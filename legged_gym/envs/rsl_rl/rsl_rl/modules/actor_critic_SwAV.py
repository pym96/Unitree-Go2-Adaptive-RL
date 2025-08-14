import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
import torch.nn.functional as F 
import os

#---------------------------TSNE-----------------------------------------------
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def Tsne(z):
    z = z.detach().cpu().numpy()

    # 用 t-SNE 降维到 2D
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    z_2d = tsne.fit_transform(z)
    return z_2d  # 返回降维后的数据

#########locomotionNP3O####################################

class ActorCriticSwAV(nn.Module):
    is_recurrent = False
    def __init__(self,  num_prop,
                        num_scan,
                        num_critic_obs,
                        num_priv_latent, 
                        num_hist,
                        num_actions,
                        scan_encoder_dims=[256, 256, 256],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        activation='elu',
                        init_noise_std=1.0,
                        **kwargs):
        super(ActorCriticSwAV, self).__init__()

        self.kwargs = kwargs
        priv_encoder_dims= kwargs['priv_encoder_dims']
        cost_dims = kwargs['num_costs']
        activation = get_activation(activation)
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_priv_latent = num_priv_latent
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0

        # n_proprio + n_scan + history_len*n_proprio + n_priv_latent
        self.num_obs = num_prop + num_scan + num_hist * num_prop + num_priv_latent
        self.obs_normalize = EmpiricalNormalization(self.num_obs)

        self.teacher_act = kwargs['teacher_act']
        if self.teacher_act:
            print("ppo with teacher actor")
        else:
            print("ppo with teacher actor")

        self.imi_flag = kwargs['imi_flag']
        if self.imi_flag:
            print("run imitation")
        else:
            print("no imitation")
#特权信息编码
        if len(priv_encoder_dims) > 0:
            priv_encoder_layers = mlp_factory(activation,num_priv_latent,None,priv_encoder_dims,last_act=True)
            self.priv_encoder = nn.Sequential(*priv_encoder_layers)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent
##地形高度编码 not using scan encoder,因为内部是空的
        if self.if_scan_encode:
            # scan_encoder_layers = mlp_factory(activation,num_scan,None,scan_encoder_dims,last_act=True)
            # self.scan_encoder = nn.Sequential(*scan_encoder_layers)
            # self.scan_encoder_output_dim = scan_encoder_dims[-1]
            scan_encoder_layers = mlp_factory(activation,num_scan,scan_encoder_dims[-1],scan_encoder_dims[:-1],last_act=False)
            self.scan_encoder = nn.Sequential(*scan_encoder_layers)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            print("not using scan encoder")
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan
#历史信息编码
        self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist, 16)

        # #MlpBarlowTwinsActor
        self.actor_teacher_backbone = MlpSinkhornActor(num_prop=num_prop-3,
                                      num_hist=5,#5
                                      num_actions=num_actions,
                                      actor_dims=[512,256,128],
                                      mlp_encoder_dims=[128,64],
                                      activation=activation,
                                      latent_dim=16,
                                      obs_encoder_dims=[128,64])
     
        print(self.actor_teacher_backbone)

        # Value function
        critic_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim,1,critic_hidden_dims,last_act=False)
        self.critic = nn.Sequential(*critic_layers)

        # cost function
        cost_layers = mlp_factory(activation,num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim,cost_dims,critic_hidden_dims,last_act=False)
        cost_layers.append(nn.Softplus())
        self.cost = nn.Sequential(*cost_layers)

        # Action noise
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
        
    def set_teacher_act(self,flag):
        self.teacher_act = flag
        if self.teacher_act:
            print("acting by teacher")
        else:
            print("acting by student")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    def get_std(self):
        return self.std
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        mean = self.act_teacher(obs)
        self.distribution = Normal(mean, mean*0. + self.get_std())
      

    def act(self, obs,**kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        mean = self.act_teacher(obs)
        return  mean
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_teacher(self,obs, **kwargs):
        # obs_prop = obs[:, :self.num_prop]
        # obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        obs_prop = obs[:,3 :self.num_prop]
        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)[:,:,3:]
        # print("obs_prop:")
        # print(obs_prop)  # 直接打印 PyTorch 张量
        # print("obs_hist:")
        # print(obs_hist)  # 直接打印 PyTorch 张量
        mean = self.actor_teacher_backbone(obs_prop,obs_hist)
        return mean
        
    def evaluate(self, obs, **kwargs):
        obs = self.obs_normalize(obs)

        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        #history_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent], dim=1)
        value = self.critic(backbone_input)
        return value
    
    def evaluate_cost(self,obs, **kwargs):
        obs = self.obs_normalize(obs)

        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        #history_latent = self.infer_hist_latent(obs)

        backbone_input = torch.cat([obs_prop,latent,scan_latent], dim=1)
        value = self.cost(backbone_input)
        return value
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def imitation_learning_loss(self, obs,imi_weight=1):
        # obs_prop = obs[:, :self.num_prop]
        # obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        obs_prop = obs[:, 3:self.num_prop]
        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        # contact = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + 4]
        # vel = obs_hist[:,-1,:3]

        # priv = torch.cat([contact,vel],dim=-1)
        priv = obs_hist[:,-1,:3]

        loss = self.actor_teacher_backbone.SinkhornLoss(obs_prop,obs_hist[:,:,3:],priv,5e-3)
        # loss = self.actor_teacher_backbone.SimSiamLoss(obs_prop,obs_hist[:,:,3:],priv,scan)
        # loss = self.actor_teacher_backbone.VaeLoss(obs_prop,obs_hist[:,:,3:],priv,scan)
        # loss = self.actor_teacher_backbone.VaeLoss(obs_prop,obs_hist[:,:,3:],priv)
        #loss = self.actor_teacher_backbone.maeLoss(obs_prop,obs_hist,priv)
        # loss = recon_loss + kl_loss + mseloss
        return loss
    
    def imitation_mode(self):
        pass
    
    def save_torch_jit_policy(self,path,device):
        self.actor_teacher_backbone.eval()

        obs_demo_input = torch.randn(1,self.num_prop-3).half().to(device)
        hist_demo_input = torch.randn(1,self.num_hist,self.num_prop-3).half().to(device)
        model_jit = torch.jit.trace(self.actor_teacher_backbone,(obs_demo_input,hist_demo_input))
        model_jit.save(path)

    def export_to_pt(self, path):
        # 切换到评估模式
        self.actor_teacher_backbone.eval()
        self.actor_teacher_backbone.to("cuda:0")
        # 创建保存目录（如果不存在）
        os.makedirs(path, exist_ok=True)
        
        # 定义保存路径
        path = os.path.join(path, 'actor.pt')

        traced_script_module = torch.jit.script(self.actor_teacher_backbone)
        traced_script_module.save(path)
        
        print(f"Model saved to {path}")
    def export_to_onnx(self,path):
        self.actor_teacher_backbone.eval()
        os.makedirs(path,exist_ok=True)
        path=os.path.join(path,'actor.onnx')
        dummy_obs=torch.randn(1,self.num_prop-3,device="cuda:0")
        dummy_history_obs=torch.randn(1,self.num_hist,self.num_prop-3,device="cuda:0")
        torch.onnx.export(self.actor_teacher_backbone,
                          (dummy_obs,dummy_history_obs),
                          path,
                          export_params=True,
                          opset_version=11,
                          input_names=["obs","history_obs"],
                          output_names=["actions"],
                          dynamic_axes={
                              "obs":{0:"batch_size"},
                              "history_obs":{0:"batch_size"},
                              "actions":{0:"batch_size"}
                          })
        

class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.count = 0

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values.

        Args:
            x (ndarray or Variable): Input values

        Returns:
            ndarray or Variable: Normalized output values
        """

        if self.training:
            self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""

        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count

        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean

class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False,final_act=True):
        # self.device = device
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10
        # last_activation = nn.ELU()

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        else:
            raise(ValueError("tsteps must be 10, 20 or 50"))

        if final_act:
            self.linear_output = nn.Sequential(
                nn.Linear(channel_size * 3, output_size), self.activation_fn
                )
        else:
            self.linear_output = nn.Sequential(nn.Linear(channel_size * 3, output_size))

    def forward(self, obs):
        # nd * T * n_proprio
        nd = obs.shape[0]
        T = self.tsteps
        # print("obs device", obs.device)
        # print("encoder device", next(self.encoder.parameters()).device)
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output
    

class MlpSinkhornActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 obs_encoder_dims,
                 mlp_encoder_dims,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpSinkhornActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.obs_normalizer = EmpiricalNormalization(shape=num_prop)
        self.proto = nn.Embedding(32,16)
        self.temperature= 3.0

        self.mlp_encoder = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=num_prop*num_hist,
                                 out_dims=None,
                                 hidden_dims=mlp_encoder_dims))
        self.latent_layer = nn.Sequential(nn.Linear(mlp_encoder_dims[-1],32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        
        # self.mlp_encoder = nn.Sequential(*mlp_factory(activation=activation,
        #                          input_dims=num_prop*num_hist,
        #                          out_dims=None,
        #                          hidden_dims=mlp_encoder_dims))
        # self.latent_layer = nn.Sequential(nn.Linear(mlp_encoder_dims[-1],32),
        #                                 #   nn.BatchNorm1d(32),
        #                                   nn.ELU(),
        #                                   nn.Linear(32,latent_dim))
        


        self.vel_layer = nn.Linear(mlp_encoder_dims[-1],3)

        self.actor = nn.Sequential(*mlp_factory(activation=activation,
                                 input_dims=latent_dim + num_prop + 3,
                                 out_dims=num_actions,
                                 hidden_dims=actor_dims))


        
        self.projector = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
                                 input_dims=latent_dim,
                                 out_dims=64,
                                 hidden_dims=[64],
                                 bias=False))
        
        # self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist*2, 3,final_act=False)
        
        self.bn = nn.BatchNorm1d(64,affine=False)

    def normalize(self,obs,obs_hist):
        obs = self.obs_normalizer(obs)
        obs_hist = self.obs_normalizer(obs_hist.reshape(-1,self.num_prop)).reshape(-1,10,self.num_prop)
        return obs,obs_hist

    def forward(self,obs,obs_hist):
        obs,obs_hist = self.normalize(obs,obs_hist)
        # with torch.no_grad():
        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        b,_,_ = obs_hist_full.size()
        # obs_hist_full = obs_hist_full[:,5:,:].view(b,-1)
        with torch.no_grad():
            latent = self.mlp_encoder(obs_hist_full[:,5:,:].reshape(b,-1))#obs_hist_full[:,5:,:]
            z = self.latent_layer(latent)
            vel = self.vel_layer(latent)
            # vel = self.history_encoder(obs_hist_full).detach()
            # #z = F.normalize(latents[:,3:],dim=-1,p=2).detach()
            # z = latents[:,3:].detach()
            # vel = latents[:,:3].detach()
        # self.tsne_counter += 1
        # if self.tsne_counter % self.tsne_interval == 0:
        #     plot_tsne(z)
        actor_input = torch.cat([vel.detach(),z.detach(),obs.detach()],dim=-1)
        mean  = self.actor(actor_input)
        # mean = self.actor(torch.cat([vel.detach(),z.detach()],dim=-1),obs.detach())
        return mean
    
    

    def SinkhornLoss(self,obs,obs_hist,priv,weight):
        obs,obs_hist = self.normalize(obs,obs_hist)

        obs = obs.detach()
        obs_hist = obs_hist.detach()

        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        b = obs.size()[0]

        # obs_hist = obs_hist[:,5:,:].reshape(b,-1)

        z1 = self.mlp_encoder(obs_hist_full[:,5:,:].reshape(b,-1))##obs_hist_full[:,5:,:]
        z2 = self.mlp_encoder(obs_hist[:,5:,:].reshape(b,-1))#obs_hist[:,5:,:]

        z1_l = self.latent_layer(z1)
        z1_v = self.vel_layer(z1)

        z2_l = self.latent_layer(z2)
        # z2_v = z2[:,:3]

        z1_l = F.normalize(z1_l,dim=-1,p=2)
        z2_l = F.normalize(z2_l,dim=-1,p=2)


        with torch.no_grad():
            w = self.proto.weight.data.clone()
            w = F.normalize(w, dim=-1, p=2)
            self.proto.weight.copy_(w)

        score_s = z1_l @ self.proto.weight.T#输出32
        score_t = z2_l @ self.proto.weight.T

        with torch.no_grad():
            q_s = sinkhorn(score_s)
            q_t = sinkhorn(score_t)

        log_p_s = F.log_softmax(score_s / self.temperature, dim=-1)
        log_p_t = F.log_softmax(score_t / self.temperature, dim=-1)

        swap_loss = -0.5 * (q_s * log_p_t + q_t * log_p_s).mean()
        priv_loss = F.mse_loss(z1_v,priv)#默认对整个batch求平均值
        losses =priv_loss + swap_loss

        # self.optimizer.zero_grad()
        # losses.backward()
        # nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        # self.optimizer.step()
        return losses   
    
###########################################################################
#####################################################


@torch.no_grad()
def sinkhorn(out, eps=0.05, iters=3):
    Q = torch.exp(out / eps).T
    K, B = Q.shape[0], Q.shape[1]
    Q /= Q.sum()

    for it in range(iters):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    return (Q * B).T

def off_diagonal(x):
    n,m = x.shape
    assert n==m
    return x.flatten()[:-1].view(n-1,n+1)[:,1:].flatten()
 
def mlp_factory(activation, input_dims, out_dims, hidden_dims,last_act=False):
    layers = []
    layers.append(nn.Linear(input_dims, hidden_dims[0]))
    layers.append(activation)
    for l in range(len(hidden_dims)-1):
        layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
        layers.append(activation)

    if out_dims:
        layers.append(nn.Linear(hidden_dims[-1], out_dims))
    if last_act:
        layers.append(activation)

    return layers

def mlp_batchnorm_factory(activation, input_dims, out_dims, hidden_dims,last_act=False,bias=True):
    layers = []
    layers.append(nn.Linear(input_dims, hidden_dims[0],bias=bias))
    layers.append(nn.BatchNorm1d(hidden_dims[0]))
    layers.append(activation)
    for l in range(len(hidden_dims)-1):
        layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1],bias=bias))
        layers.append(nn.BatchNorm1d(hidden_dims[l + 1]))
        layers.append(activation)

    if out_dims:
        layers.append(nn.Linear(hidden_dims[-1], out_dims,bias=bias))
    if last_act:
        layers.append(activation)

    return layers

    
def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None

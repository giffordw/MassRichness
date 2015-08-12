import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from funcs_plotting import *
plt.style.use('ggplot')

#########################
#Read in and define data#
#########################

#size of background relative to cluster aperature
c = 5.0
#degrees of freedom
nu = 6
#import data
dt =  [('m200',float),('obstot1',int),('obsbkg1',float)]
data1 = np.loadtxt('DataSet3_C4_Halos_rev.dat', dtype=dt, usecols=(0,1,2))
#select only good richness values
data1 = data1[data1['obstot1']-data1['obsbkg1']/c>0]
N = len(data1['m200'])
#scale mass values
data1['m200'] = data1['m200']*0.7/1e14
lgdy2 = (np.ones(N)*0.35)**2
lgy = np.log(data1['m200']*1e14/0.7)
#set richness values
rich = data1['obstot1']-data1['obsbkg1']/c
richtot = data1['obstot1']
richbk = data1['obsbkg1']



#Model starts HERE
#-----------------

##############
#model priors#
##############

alpha = pm.Normal("alpha",0,.0001,value=0.0) #intercept prior
beta = pm.NoncentralT("beta",1,1,10,value=1.1) #slope prior


#################
#richness priors#
#################

n200 = pm.Uniform("n200",0,3000,size=N,value=rich)
@pm.deterministic
def lgn200(n200=n200):
    return np.log(n200)
nbkg = pm.DiscreteUniform("nbkg",0,3000,size=N,value=richbk)
obsbkg = pm.Poisson("obsbkg",nbkg,observed=True,value=richbk)
obstot = pm.Poisson("obstot",nbkg/c + n200,observed=True,value=richtot)


#############
#Mass priors#
#############

#### mass scatter priors
scat_alpha = pm.Uniform('scat_alpha',lower=0.001,upper=0.5,value=0.1)
scat_beta = pm.Uniform('scat_beta',lower=0.01,upper=1.0,value=0.1)
@pm.deterministic
def intrscat_std(scat_alpha=scat_alpha,scat_beta=scat_beta,lgn200=lgn200):
    return scat_alpha + scat_beta/(1.0+lgn200)
#intrscat_std = pm.Beta('intrscat_std',0.15*20,(1-0.15)*20)
#intrscat_std = 0.0
#prec_intrscat = 1.0 / intrscat_std**2.0
precy_std = pm.Uniform('precy_std',0.001,0.7,size=N)
precy = 1.0 / (precy_std**2.0 + intrscat_std**2.0)

#### mass priors
#lgM200 = pm.Normal("lgM200",alpha + 14.5 + beta*(lgn200 - 1.5),0.05)
lgM200 = alpha+14.5+beta*(lgn200 - 1.5)
#Mass models
obs_var_lgm200 = pm.Beta('obs_var_lgm200',precy_std**2*1000.0,(1.0-precy_std**2)*1000.0,observed=True,value=lgdy2)
#obsvarlgm200 = lgdy2
#obslgM200 = pm.Normal("obslgM200",lgM200,precy,observed=True,value=lgy) #model assuming no outliers



###################
#likelihood priors#
###################

# uniform prior on Pb, the fraction of bad points
#Pb = pm.Uniform('Pb', 0, 1.0, value=0.1)
Pb = 0.0

# uniform prior on Yb, the centroid of the outlier distribution
Yb = pm.Uniform('Yb', 25, 40,value=33)

# uniform prior on log(sigmab), the spread of the outlier distribution
log_sigmab = pm.Uniform('log_sigmab', -1, 10, value=5)
@pm.deterministic
def sigmab(log_sigmab=log_sigmab):
    return np.exp(log_sigmab)

#Good vs. Bad parameter for all data points
qi = pm.Bernoulli('qi', p=1 - Pb, value=np.random.rand(len(lgy)))



#####################
#Likelihood Function#
#####################
def outlier_likelihood(yi, mu, Yb, dyi, qi,sigmab,int_scat):
    """likelihood for full outlier posterior"""
    Vi = dyi**2
    Vb = sigmab ** 2
    logL_in = -0.5 * np.sum(qi * (np.log(2 * np.pi * (Vi+int_scat**2)) + (yi - mu) ** 2 / (Vi+int_scat**2)))
    logL_out = -0.5 * np.sum((1 - qi) * (np.log(2 * np.pi * (Vi + Vb)) + (yi - Yb) ** 2 / (Vi + Vb)))
    return logL_out + logL_in



############
#PyMC stuff#
############
OutlierNormal = pm.stochastic_from_dist('outliernormal',logp=outlier_likelihood,dtype=np.float,mv=True)
y_outlier = OutlierNormal('y_outlier', mu=lgM200, dyi=precy_std, Yb=Yb,sigmab=sigmab, qi=qi,int_scat=intrscat_std, observed=True, value=lgy)
mcmc = pm.MCMC([alpha,beta,n200,lgn200,nbkg,obsbkg,obstot,lgM200,y_outlier,obs_var_lgm200,precy_std,log_sigmab,sigmab,Yb,qi,intrscat_std,scat_alpha,scat_beta])
mcmc.sample(iter=200000, burn=100000)
pymc_trace = [mcmc.trace('alpha')[:],
              mcmc.trace('beta')[:],
              1.0/mcmc.trace('intrscat_std')[:]**2]


plot_MCMC_results(np.log(rich), lgy, pymc_trace)
plt.show()

plt.plot(np.log(rich), lgy,'ko')
plt.errorbar(np.log(rich), lgy,xerr=0.434/np.sqrt(rich),yerr=np.median(mcmc.trace('precy_std')[:],0),fmt='None',ecolor='darkgrey')
xgrid = np.arange(0.0,5.0,0.01)
model_values = np.median(pymc_trace[0]) + 14.5 + np.median(pymc_trace[1])*(xgrid - 1.5)
plt.plot(xgrid,model_values,'k',lw=2)
#int_scat = np.median(mcmc.trace('intrscat_std')[:],0)
int_scat = np.median(mcmc.trace('scat_alpha')[:]) + np.median(mcmc.trace('scat_beta')[:])/(1+xgrid)
#plt.annotate('Intrinsic Scatter:%0.2f'%(int_scat),(0.6,15.7),(0.6,15.7),fontsize=15)
plt.fill_between(xgrid,model_values+int_scat,model_values-int_scat,facecolor='grey',alpha=0.3)
Pi = mcmc.trace('qi')[:].mean(0)
plt.plot(np.log(rich)[Pi<0.1], lgy[Pi<0.1],'ro')
rand = np.random.uniform(0,100000,100)
for i in rand:
    plt.plot(np.arange(0.5,2.5,0.01),mcmc.trace('alpha')[:][i] + 14.5 + mcmc.trace('beta')[:][i]*(np.arange(0.5,2.5,0.01) - 1.5),'k',alpha=0.05,lw=.5)
#plt.plot(np.arange(0,10),-0.13 + 14.5 + np.median(mcmc.trace('beta')[:])*(np.arange(0,10) - 1.5))
plt.xlabel('ln(N200)',fontsize=16)
plt.ylabel('ln(M200)',fontsize=16)
plt.xlim(0.0,5.0)
plt.ylim(30,36)
plt.show()


import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
plt.style.use('ggplot')

#########
# Create some convenience routines for plotting
#########

def compute_sigma_level(trace1, trace2, nbins=20):
    """From a set of traces, bin by number of standard deviations"""
    L, xbins, ybins = np.histogram2d(trace1, trace2, nbins)
    L[L == 0] = 1E-16
    logL = np.log(L)

    shape = L.shape
    L = L.ravel()

    # obtain the indices to sort and unsort the flattened array
    i_sort = np.argsort(L)[::-1]
    i_unsort = np.argsort(i_sort)

    L_cumsum = L[i_sort].cumsum()
    L_cumsum /= L_cumsum[-1]
    
    xbins = 0.5 * (xbins[1:] + xbins[:-1])
    ybins = 0.5 * (ybins[1:] + ybins[:-1])

    return xbins, ybins, L_cumsum[i_unsort].reshape(shape)


def plot_MCMC_trace(ax, xdata, ydata, trace, scatter=False, **kwargs):
    """Plot traces and contours"""
    xbins, ybins, sigma = compute_sigma_level(trace[0], trace[1])
    ax.contour(xbins, ybins, sigma.T, levels=[0.683, 0.955], **kwargs)
    if scatter:
        ax.plot(trace[0], trace[1], ',k', alpha=0.1)
    plt.plot([np.median(trace[0])],[np.median(trace[1])],'ro',ms=8)
    ax.set_xlabel(r'$\alpha$',fontsize=15)
    ax.set_ylabel(r'$\beta$',fontsize=15)
    
    
def plot_MCMC_model(ax, xdata, ydata, trace):
    """Plot the linear model and 2sigma contours"""
    ax.plot(xdata, ydata, 'ok')

    alpha, beta = trace[:2]
    xfit = np.linspace(0.8, 2.3, 10)
    yfit = alpha[:, None]+ 14.5 + beta[:, None] * (xfit - 1.5)
    mu = np.median(yfit,0)
    sig = 2 * yfit.std(0)

    ax.plot(xfit, mu, '-k')
    ax.fill_between(xfit, mu - sig, mu + sig, color='lightgray')

    ax.set_xlabel('x')
    ax.set_ylabel('y')


def plot_MCMC_results(xdata, ydata, trace, colors='k'):
    """Plot both the trace and the model together"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    plot_MCMC_trace(ax, xdata, ydata, trace, True, colors=colors)
    #plot_MCMC_model(ax[1], xdata, ydata, trace)

#Read in and define data
c = 5.0
nu = 6 #degrees of freedom
dt =  [('m200',float),('obstot1',int),('obsbkg1',float)]
data1 = np.loadtxt('DataSet3_C4_Halos_rev.dat', dtype=dt, usecols=(0,1,2))
data1 = data1[data1['obstot1']-data1['obsbkg1']/c>0]
N = len(data1['m200'])
data1['m200'] = data1['m200']*0.7/1e14
#lgdy2 = (np.log(data1['m200']*1e14/0.7) - np.log((m200e + data1['m200'])*1e14/0.7))**2
lgdy2 = (np.ones(N)*0.35)**2
lgy = np.log(data1['m200']*1e14/0.7)


###################
#Model starts HERE#
###################
#define model priors
alpha = pm.Normal("alpha",0,.0001,value=0.0) #intercept prior
beta = pm.NoncentralT("beta",1,1,10,value=1.1) #slope prior

#Richness priors
n200 = pm.Uniform("n200",0,3000,size=N,value=data1['obstot1']-data1['obsbkg1']/c)
@pm.deterministic
def lgn200(n200=n200):
    return np.log(n200)
nbkg = pm.DiscreteUniform("nbkg",0,3000,size=N,value=data1['obsbkg1'])
#Richness models
obsbkg = pm.Poisson("obsbkg",nbkg,observed=True,value=data1['obsbkg1'])
obstot = pm.Poisson("obstot",nbkg/c + n200,observed=True,value=data1['obstot1'])

#Mass priors
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
#lgM200 = pm.Normal("lgM200",alpha + 14.5 + beta*(lgn200 - 1.5),0.05)
lgM200 = alpha+14.5+beta*(lgn200 - 1.5)
#Mass models
obs_var_lgm200 = pm.Beta('obs_var_lgm200',precy_std**2*1000.0,(1.0-precy_std**2)*1000.0,observed=True,value=lgdy2)
#obsvarlgm200 = lgdy2
#obslgM200 = pm.Normal("obslgM200",lgM200,precy,observed=True,value=np.log(data1['m200']*1e14/0.7)) #model assuming no outliers


#####
#Testing outlier values
#####
# uniform prior on Pb, the fraction of bad points
Pb = pm.Uniform('Pb', 0, 1.0, value=0.1)
# uniform prior on Yb, the centroid of the outlier distribution
Yb = pm.Uniform('Yb', 25, 40,value=33)
# uniform prior on log(sigmab), the spread of the outlier distribution
log_sigmab = pm.Uniform('log_sigmab', -1, 10, value=5)
@pm.deterministic
def sigmab(log_sigmab=log_sigmab):
    return np.exp(log_sigmab)
#try building outlier model
qi = pm.Bernoulli('qi', p=1 - Pb, value=np.random.rand(len(lgy)))
#qi = np.ones(len(lgy))

def outlier_likelihood(yi, mu, Yb, dyi, qi,sigmab,int_scat):
    """likelihood for full outlier posterior"""
    Vi = dyi**2
    Vb = sigmab ** 2
    root2pi = np.sqrt(2 * np.pi)
    logL_in = -0.5 * np.sum(qi * (np.log(2 * np.pi * (Vi+int_scat**2)) + (yi - mu) ** 2 / (Vi+int_scat**2)))
    logL_out = -0.5 * np.sum((1 - qi) * (np.log(2 * np.pi * (Vi + Vb)) + (yi - Yb) ** 2 / (Vi + Vb)))
    #L_in = (1. / root2pi / dyi * np.exp(-0.5 * (yi - mu) ** 2 / Vi))
    #L_out = (1. / root2pi / np.sqrt(Vi + Vb) * np.exp(-0.5 * (yi - Yb) ** 2 / (Vi + Vb)))
    return logL_out + logL_in
    #return np.sum(np.log((1 - Pb) * L_in + Pb * L_out))

OutlierNormal = pm.stochastic_from_dist('outliernormal',logp=outlier_likelihood,dtype=np.float,mv=True)
y_outlier = OutlierNormal('y_outlier', mu=lgM200, dyi=precy_std, Yb=Yb,sigmab=sigmab, qi=qi,int_scat=intrscat_std, observed=True, value=lgy)
#MCMC sampler
mcmc = pm.MCMC([alpha,beta,n200,lgn200,nbkg,obsbkg,obstot,lgM200,y_outlier,obs_var_lgm200,precy_std,log_sigmab,sigmab,Yb,Pb,qi,intrscat_std,scat_alpha,scat_beta])



#MCMC sampler
#mcmc = pm.MCMC([alpha,beta,n200,lgn200,nbkg,obsbkg,obstot,lgM200,obslgM200,obs_var_lgm200,precy_std,intrscat_std,precy,scat_alpha,scat_beta])
mcmc.sample(iter=200000, burn=100000)

pymc_trace = [mcmc.trace('alpha')[:],
              mcmc.trace('beta')[:],
              1.0/mcmc.trace('intrscat_std')[:]**2]

plot_MCMC_results(np.log(data1['obstot1'] - data1['obsbkg1']/c), np.log(data1['m200']*1e14/0.7), pymc_trace)
plt.show()
plt.plot(np.log(data1['obstot1'] - data1['obsbkg1']/c), np.log(data1['m200']*1e14/0.7),'ko')
plt.errorbar(np.log(data1['obstot1'] - data1['obsbkg1']/c), np.log(data1['m200']*1e14/0.7),xerr=0.434/np.sqrt(data1['obstot1'] - data1['obsbkg1']/c),yerr=np.median(mcmc.trace('precy_std')[:],0),fmt='None',ecolor='darkgrey')
xgrid = np.arange(0.0,5.0,0.01)
model_values = np.median(mcmc.trace('alpha')[:]) + 14.5 + np.median(mcmc.trace('beta')[:])*(xgrid - 1.5)
plt.plot(xgrid,model_values,'k',lw=2)
#int_scat = np.median(mcmc.trace('intrscat_std')[:],0)
int_scat = np.median(mcmc.trace('scat_alpha')[:]) + np.median(mcmc.trace('scat_beta')[:])/(1+xgrid)
#plt.annotate('Intrinsic Scatter:%0.2f'%(int_scat),(0.6,15.7),(0.6,15.7),fontsize=15)
plt.fill_between(xgrid,model_values+int_scat,model_values-int_scat,facecolor='grey',alpha=0.3)
Pi = mcmc.trace('qi')[:].mean(0)
plt.plot(np.log(data1['obstot1'] - data1['obsbkg1']/c)[Pi<0.2], np.log(data1['m200']*1e14/0.7)[Pi<0.2],'ro')
rand = np.random.uniform(0,100000,100)
for i in rand:
    plt.plot(np.arange(0.5,2.5,0.01),mcmc.trace('alpha')[:][i] + 14.5 + mcmc.trace('beta')[:][i]*(np.arange(0.5,2.5,0.01) - 1.5),'k',alpha=0.05,lw=-.5)
#plt.plot(np.arange(0,10),-0.13 + 14.5 + np.median(mcmc.trace('beta')[:])*(np.arange(0,10) - 1.5))
plt.xlabel('ln(N200)',fontsize=16)
plt.ylabel('ln(M200)',fontsize=16)
plt.xlim(0.0,5.0)
plt.ylim(30,36)
plt.show()


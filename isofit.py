import numpy as np
import pandas as pd

import sys
from os import path
from itertools import product

#You shouldn't need these lines unless you make your own model grid. But who knows
#from isochrones.mist import MIST_EvolutionTrack, MISTEvolutionTrackGrid
#from isochrones.mist.bc import MISTBolometricCorrectionGrid


from scipy.optimize import minimize 

from .interp import DFInterpolator
import emcee


class SingleStar(object):

    def __init__(self, obs, ModelObject, use_course_grid=True):

        models = ModelObject.models

        self.obs = obs # A Dictionary of Value, Err to fit to the models. 
        self.obs_keys = list( self.obs.keys() )
        self.mag_keys = [k for k in self.obs_keys if k[-3:]=='mag']

        self.StarModels = ModelObject

        if use_course_grid:
            self.all_models = self._make_course_grid()
        else:
            self.all_models = models[~np.isnan(models[self.mag_keys[0]])]
            
        self.good_models = self._cut_badfit_models()


        self.mcmc_posterior = None
        self.physical_posterior = None
        self.posterior_models = None
        

    def _make_course_grid(self, params=None):

        if params is None:
            test_masses = np.arange(0.2, 8., 0.05)
            test_eep = np.arange(205, 1700, 5.)

            if self._in_obs('feh'):
                test_feh=[self.obs['feh'][0]]
            else:
                test_feh = np.arange(-1, 1, 0.1)
        
            inputs = np.array(list(product(test_masses, test_eep, test_feh,[100.], [0.])))

        else:
            inputs=params
            
        course_grid = self.StarModels.interpolate(inputs.T)

        return course_grid.dropna()

        

    def _update_obs(self, obs, sig=5.):
        self.obs = obs
        self.obs_keys= list( self.obs.keys() )
        self.mag_keys = [k for k in self.obs_keys if k[-3:]=='mag']
        self.all_models = self._make_course_grid()
        self.good_models = self._cut_badfit_models(sig=5.)
        



    def _cut_badfit_models(self, sig=3., cutmag=False):

        model_cut  = np.ones(len(self.all_models), dtype=bool )
        models = self.all_models

        # First cut non-reddened quantities
        if self._in_obs('Teff'):

            teff_max = self.obs['Teff'][0] + sig * self.obs['Teff'][1]
            teff_min = self.obs['Teff'][0] - sig * self.obs['Teff'][1]
            model_cut &= models['Teff'] > teff_min
            model_cut &= models['Teff'] < teff_max

            print('{} Models remaining after Teff Cuts'.format(np.sum(model_cut)))

        
        if self._in_obs('feh'):
            feh_max = self.obs['feh'][0] + sig * self.obs['feh'][1]
            feh_min = self.obs['feh'][0] - sig * self.obs['feh'][1]
            model_cut &= models['feh'] > feh_min
            model_cut &= models['feh'] < feh_max

            print('{} Models remaining after feh cuts'.format(np.sum(model_cut)))

        
        if self._in_obs('logg'):

            logg_max = self.obs['logg'][0] + sig * self.obs['logg'][1]
            logg_min = self.obs['logg'][0] - sig * self.obs['logg'][1]
            model_cut &= models['logg'] > logg_min
            model_cut &= models['logg'] < logg_max

            print('{} Models remaining after logg Cuts'.format(np.sum(model_cut)))
        

        models = models[model_cut]
        model_cut = np.ones(len(models), dtype=bool)

        if not(cutmag):
            if np.sum(model_cut)==0.:
                print('BEWARE: NO GOOD MODELS. USING ALL MODELS.')
                return self.all_models.loc[~np.isnan(self.all_models['2MASS_Ks_mag'])]
            return models.loc[~np.isnan(models['2MASS_Ks_mag'])]

        
        # Cut on Magnitudes/Distance:
        if self._in_obs('plax'):

            plax_min = self.obs['plax'][0] - sig * self.obs['plax'][1]
            plax_max = self.obs['plax'][0] + sig * self.obs['plax'][1]
            
            d_mod_min = 5. * np.log10(1000./plax_max) - 5.
            d_mod_max = 5. * np.log10(1000./plax_min) - 5.
            
            if  np.isin(['ebv'], self.obs_keys)[0]:
                ebv, ebv_err = self.obs['ebv']
            else:
                ebv, ebv_err = 0., 0.

            for magkey in self.mag_keys:
                
                mag, magerr = self.obs[magkey]

                dered = extinction(magkey, ebv)

                dered_mag = mag - dered
                dered_magerr = (magerr**2. + (dered*ebv_err)**2. )**0.5
                
                abs_mag_min = dered_mag - sig*dered_magerr - d_mod_min 
                abs_mag_max = dered_mag + sig*dered_magerr - d_mod_max

                model_cut &= models[magkey] > abs_mag_min
                model_cut &= models[magkey] < abs_mag_max

        if np.sum(model_cut)==0.:
            print('BEWARE: NO GOOD MODELS. USING ALL MODELS.')
            return self.all_models[~np.isnan(self.all_models['2MASS_Ks_mag'])]
        else:
            print('{} Models remaining after magnitude and distance cuts'.format(np.sum(model_cut)) )
            return models[model_cut]
        

    def _in_obs(self, k):
        return  np.isin([k], self.obs_keys)[0]

    def _in_mags(self, k):
        return  np.isin([k], self.mag_keys)[0]


    def get_implied_model_parallax(self, magkey='2MASS_Ks_mag'):

        mag, magerr = self.obs[magkey]

        if self._in_obs('ebv'):

            ebv, ebv_err = self.obs['ebv']
            dered = extinction(magkey, ebv)
            dered_mag = mag - dered

        else:
            dered_mag = mag

        model_mags = self.good_models[magkey]
        model_plax = 10. ** (-0.2 * (dered_mag - model_mags + 5.) + 3. )

        return model_plax

        
    def mod_prior(self,age_prior=10.31):

        parallax = self.get_implied_model_parallax()
        
        if self._in_obs('feh'):
            prior = ln_distance_prior(1000./parallax)
        else:
            feh = self.good_models['feh']
            prior = ln_distance_prior(1000./parallax) + ln_gauss_prior(feh, 0., 0.2)        

        return prior

        
    def mod_likelihood(self):

        obs=self.obs

        tot_likelh = np.ones(len(self.good_models))
        
        for k in self.obs_keys:

            if self._in_mags(k) and self._in_obs('plax'):
                
                plax, plax_err = obs['plax']
                mag, magerr = obs[k]

                if self._in_obs('ebv'):

                    ebv, ebv_err = obs['ebv']
                    dered = extinction(k, ebv)
                    dered_mag = mag - dered
                    dered_magerr = np.sqrt(magerr**2. + (dered*ebv_err)**2. )

                else:
                    dered_mag = mag
                    dered_magerr = magerr
                
                abs_mag = -5.0 * np.log10(1000. / plax) + dered_mag + 5.0
                abs_mag_err = np.sqrt(
                    (-5.0 / (plax * np.log(10)))**2 * plax_err**2 + dered_magerr**2  )
                
                tot_likelh += ln_gauss_prior(self.good_models[k], abs_mag, abs_mag_err)
                
            elif not(k=='ebv' or k=='plax'):
                x, xerr = obs[k]
                tot_likelh += ln_gauss_prior(self.good_models[k], x, xerr)

        return tot_likelh
        

    def mod_prob(self):

        likelihood = self.mod_likelihood()
        prior = self.mod_prior()

        return prior + likelihood

    
    def return_best_model(self):

        prob = self.mod_prob()
        best_mod = self.good_models.iloc[np.argmax(np.array(prob))]
    
        best_mass, best_eep, best_feh = best_mod[['mass', 'eep', 'feh']].to_numpy()
        
        zoom_masses = np.arange(best_mass-0.1, best_mass+0.1, 0.01)
        
        zoom_eep = np.arange(best_eep-100., best_eep+150, 0.5)
        zoom_feh = np.arange(best_feh-0.1, best_feh+0.11, 0.01)

        inputs = np.array(list(product(zoom_masses, zoom_eep, zoom_feh, [100.], [0.])) )
        self.good_models = self._evaluate_model(inputs.T).dropna()
        zoom_prob = self.mod_prob()

        return self.good_models.iloc[np.argmax(np.array(zoom_prob))]



    def _evaluate_model(self, params):
        # params has shape mass, eep, feh, parallax, ebv

        if np.shape(params)[0]!=5:
            return self.StarModels.interpolate(params.T)

        return self.StarModels.interpolate(params)

        
    def mcmc_prior(self, model_param):

        mass, eep, feh, parallax, ebv = model_param

        if ebv<0.:
            return -np.inf
        if parallax<0.:
            return -np.inf

        if self._in_obs('feh'):
            return ln_distance_prior(1000./parallax)
        else:
            return ln_distance_prior(1000./parallax) + ln_gauss_prior(feh, 0., 0.25)
        

    def mcmc_likelihood(self, model_param, age_lim=10.31):
        
        mass, eep, feh, parallax, ebv = model_param
        model = self._evaluate_model(model_param)
        ln_lh = 0.

        if model['age']>age_lim:
            return -np.inf

        
        for k in self.obs_keys:

            if k == 'ebv':
                x, xerr = self.obs[k]
                if xerr < 5e-3:
                    xerr += 5e-3
                ln_lh += ln_gauss_prior(ebv, x, xerr )
            elif k== 'plax':
                x, xerr = self.obs[k]
                if xerr == 0:
                    xerr += 1e-6
                ln_lh += ln_gauss_prior(parallax, x, xerr )

            else:
                try:
                    x, xerr = self.obs[k]
                    if xerr == 0:
                        xerr += 1e-6
                    ln_lh += ln_gauss_prior(float(model[k]), x, xerr )

                except KeyError:
                    print(k+' NOT IN MODEL Evaluation Function. ')

        return ln_lh


    def mcmc_lnpost(self, model_param):

        likelihood = self.mcmc_likelihood(model_param)
        prior = self.mcmc_prior(model_param)

        if np.isnan(likelihood):
            return -np.inf 

        return likelihood + prior



    def run_mcmc_fit(self, nwalkers=50, nsteps=300, ndiscard=100, progress=False):

        best_model = self.return_best_model()

        mass, eep, feh = best_model[['mass', 'eep', 'feh']].to_numpy()
        if self._in_obs('ebv'):
            plax, ebv = self.obs['plax'][0], self.obs['ebv'][0]
        else:
            ebv=0.
            plax = self.obs['plax'][0]
        
        init_samples = [mass, eep, feh, plax, ebv]
        ndim=len(init_samples)
        
        init_samples = np.array([s + np.random.normal(0, 1e-3, size=nwalkers) for s in init_samples]).T

        init_samples[4] = np.abs(init_samples[4])
        init_samples[3] = np.abs(init_samples[3])

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.mcmc_lnpost )
        sampler.run_mcmc(init_samples, nsteps, progress=progress, )

        posterior = sampler.get_chain(discard=ndiscard, flat=True)
        post_df = pd.DataFrame(posterior, columns=['mass', 'eep', 'feh', 'plax','ebv'])

        self.mcmc_posterior = post_df
        self.posterior_models = self._evaluate_model(posterior)
        
        return post_df.copy()



    def get_physical_posterior(self, props=['Teff','logg','feh','mass', 'radius','logL','density','age']):

        post =  self.posterior_models[props]

        post.loc[:,'distance'] = 1000./self.mcmc_posterior['plax']
        post.loc[:,'ebv'] = self.mcmc_posterior['ebv']
        post.loc[:,'eep'] = self.mcmc_posterior['eep']
        
        
        self.physical_posterior = post

        return post.copy()


    def return_fit_errs(self):

        post = self.physical_posterior
        params = dict()

        if self.physical_posterior is None:
            phys_post = self.get_physical_posterior()
        else:
            phys_post = self.physical_posterior.copy()
        
        for k in phys_post.columns:

            med, lo, hi = np.nanpercentile(post[k], [50,16,84] )

            params[k] =[med]
            params[k+'_e'] = [lo-med]
            params[k+'_E'] = [hi-med]

        return pd.DataFrame(params, )


class DoubleStar(object):


    def __init__(self, obs, ModelObject):

        self.obs = obs
        self.StarModels = ModelObject


    
class MIST_Track_Models(object):

    def __init__(self, modelgrid=None, make_new=False, mags=['2MASS_Ks']):

        if modelgrid is None:
            print('..Creating Model Grid...')
            masses = np.arange(0.1, 8, 0.02)
            eep = np.arange(202, 1710, 1) #All phases from ZAMS to WD Track
            feh = np.arange(-2, 0.8, 0.05)
            self.models = get_multiindex_model_grid(masses, eep, feh, mags=mags,make_new=make_new)
        else:
            self.models = modelgrid
 
        self.keys = self.models.columns
        self.magkeys = [k for k in self.keys if k[-4:]=='_mag']
        self.valuekeys = self.keys[np.in1d(self.keys, self.magkeys, invert=True)]

        self.extinction_law = np.array([extinction(k, 1.) for k in self.magkeys])

        dirpath = path.dirname(__file__)


        self._interp = DFInterpolator(self.models, recalc=make_new, is_full=True,
                                      filename=dirpath+'/misttrack_df_interp',)
        

    def _interp_values(self, pars):
        mass, eep, feh, plax, ebv = pars
        return self._interp([mass,eep,feh], self.valuekeys)

    def _interp_models(self, pars):
        mass, eep, feh = pars
        return self._interp([mass,eep,feh], self.valuekeys)

    def _interp_mags(self, pars):

        mass, eep, feh, plax, ebv = pars
        abs_mags = self._interp([mass,eep,feh], self.magkeys)
        
        d_mod = - 5. + 5. * np.log10(1000./plax )

        try:
            A = ebv * self.extinction_law
            rel_mags = abs_mags + d_mod + A
            
        except ValueError:
            extinction_law = np.array([self.extinction_law]*len(ebv))
            A = np.vstack(ebv) * self.extinction_law
            rel_mags = np.vstack(abs_mags) + np.vstack(d_mod) + A
            
        return rel_mags
    
    def interpolate(self, pars):

        mass, eep, feh, plax, ebv = pars

        values = self._interp_values(pars)
        mags = self._interp_mags(pars, )

        try:
            df=pd.DataFrame([np.concatenate([values, mags])], columns=self.keys)
        except ValueError:
            df=pd.DataFrame(np.concatenate([values, mags],axis=1), columns=self.keys)
        
        return df


def gauss_prior(a, mu, sig):
    return np.exp(-0.5*(a-mu)**2/sig**2)


def distance_prior(d, lscale=1350.):
    return (d**2/(2.0*lscale**3.))*np.exp(d/lscale)


def ln_gauss_prior(x, mu, sig):
    return -0.5 * (x-mu)**2. / sig**2.

def ln_distance_prior(d, lscale=1350.):
    return np.log(distance_prior(d, lscale))


def extinction( mag_key, ebv ):

    ''' EXTINCTION LAW TAKEN FROM WANG & CHEN 2019, Table 3'''

    e_gprp = 1.303 * ebv

    if mag_key=='2MASS_Ks_mag':
        A = e_gprp * 0.186

    elif mag_key =='2MASS_H_mag':
        A = e_gprp * 0.313

    elif mag_key =='2MASS_J_mag':
        A = e_gprp * 0.582

    #elif mag_key=='K_mag': # NOTE: Johnson K
    #    A = e_gprp * 0.186

    elif mag_key =='J_mag':
        A = e_gprp * 0.582

    elif mag_key =='H_mag':
        A = e_gprp * 0.313

    elif mag_key == 'RP_mag':
        A = e_gprp * 1.429

    elif mag_key == 'BP_mag':
        A = e_gprp * 2.429

    elif mag_key == 'G_mag':
        A = e_gprp * 1.890

    elif mag_key == 'W1_mag':
        A = e_gprp * 0.094

    elif mag_key == 'W2_mag':
        A = e_gprp * 0.063

    elif mag_key == 'W3_mag':
        A = e_gprp * 0.095

    elif mag_key == 'U_mag':
        A = e_gprp * 3.7665

    elif mag_key == 'B_mag':
        A = e_gprp * 3.151
        
    elif mag_key == 'V_mag':
        A = e_gprp * 2.394
        
            
    else:
        print('NO EXTINCTION COEFFICIENT FOR '+mag_key+'. ASSUMING NO EXTINCTION.')
        return 0.

    return A

    

def get_model_grid(masses, eep, feh, mags=['2MASS_Ks'], make_new=False,  h5key='mist',  fname='stellar_isochrone_models.h5'):

    fpath = path.join(path.dirname(__file__), fname)
    file_exists = path.exists(fpath)

    
    if make_new or not(file_exists):
        
        mass_grid, eep_grid, feh_grid = np.meshgrid(masses, eep, feh)

        all_mass = mass_grid.reshape((-1))
        all_eep = eep_grid.reshape((-1))
        all_feh = feh_grid.reshape((-1))


        print('...Creating Grid with {} Models...'.format(len(all_mass)) )
        
        mist_track = MIST_EvolutionTrack()

        model_tracks = mist_track( all_mass, all_eep, all_feh)
        model_tracks['in_feh'] = all_feh
        model_tracks['in_mass'] = all_mass
        model_tracks['in_eep'] = all_eep

        print('...Calculating Absolute Magnitude Grid with {} Models...'.format(len(all_mass)) )
        mag_pars = np.array([model_tracks['Teff'],model_tracks['logg'],model_tracks['feh'],np.zeros(len(model_tracks))])

        bc_grid = MISTBolometricCorrectionGrid(mags)
        
        bc = bc_grid.interp(mag_pars, mags)
        mbol = np.vstack(np.array(model_tracks['Mbol'] ) )

        mag_cols = [m+'_mag' for m in mags]
        model_mags = pd.DataFrame( mbol-bc , columns=mag_cols)

        model_tracks = model_tracks.merge(model_mags, right_index=True, left_index=True)

        print('...Saving Grid...')
        model_tracks.to_hdf(fpath, h5key, mode='w')
        
        return model_tracks
        
    else:
        
        model_tracks = pd.read_hdf(fname, h5key)
        
        return model_tracks



def get_multiindex_model_grid(mass, eep, feh, mags=['2MASS_Ks'],make_new=False,
                              key='mist', fname='stellar_mist_multiindex_modelgrid.h5'):

    fpath = path.join(path.dirname(__file__), fname)
    file_exists = path.exists(fpath)

    if make_new or not(file_exists):

         model_grid = get_model_grid(mass, eep, feh, mags=mags,make_new=make_new,h5key=key)

         #print('...{} Models Remaining after dropping NaNs ...\n...Creating Multiindex...'.format(len(model_grid)))

         #model_grid = model_grid.dropna(subset=['initial_mass','eep'])
         print('...Sorting and Organizing Grid...')
         mi_model_grid = model_grid.round(10).set_index(['in_mass','in_eep','in_feh'])
         mi_model_grid = mi_model_grid.sort_values(by=['in_mass','in_eep','in_feh'],ascending=True)

         print('...Saving Multindex Grid...')
         mi_model_grid.to_hdf(fpath, key, mode='w')

         print('Done')
         return mi_model_grid

    else:
        
        model_grid = pd.read_hdf(fpath, key)
        return model_grid



def get_isochrone_grid():

    return 1.




def get_multiindex_isochrone_grid():

    return 1.
    
    

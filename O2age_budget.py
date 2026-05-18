#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import xarray as xr
import scipy as sci
import matplotlib.pyplot as plt # still need matplotlib
import xmitgcm
import xgcm 
import gsw


# In[2]:


class AgeBudgetND(object):
    '''
    This class is designed for the budget analysis in MITgcm.
    Credit to MiniUFO and Emily 

    Here i'm improving it by trying to get the divergent terms removed... 
    '''
    def __init__(self, dset, snaps):
        '''
        Construct a Budget instance using a Dataset
        
        Parameters
        ----------
        dset : xarray.Dataset
            a given Dataset containing MITgcm output diagnostics
        
        Return
        ----------
        terms : xarray.Dataset
            A Dataset containing all budget terms
        '''
        self.grid   = xgcm.Grid(dset)
        self.coords = dset.coords.to_dataset().reset_coords()
        self.dset   = dset.reset_coords(drop=True)
        self.volume = dset.drF * dset.hFacC * dset.rA
        self.snaps  = snaps
        self.terms  = None

    ## it would be nice to just have all the terms calculated when we initialize, including the residual! 
    # maybe next coding project - should be very quick. 

    '''
    Calculate all the budget tendency terms.
    '''
    def cal_advection_tendency(self):
        '''
        Calculate tendency due to advection.
        
        Parameters
        ----------
        suffix : string
            A given string in ['TH', 'SLT'] for heat or salt calculation.
        '''
        # get MITgcm diagnostics
        ADVx = self.dset['ADVxTr11']
        ADVy = self.dset['ADVyTr11']
        ADVr = self.dset['ADVrTr11']

        # difference to get flux convergence, sign convention is opposite for verticl
        adv_x_Age = -self.grid.diff(ADVx, 'X', boundary='fill').rename('adv_x_Age')
        adv_y_Age = -self.grid.diff(ADVy, 'Y', boundary='fill').rename('adv_y_Age')
        adv_r_Age =  self.grid.diff(ADVr, 'Z', boundary='fill').rename('adv_r_Age')

        #print(adv_x_Age.name)
        
        # change unit to K/day # somehow this step lost their names!
        adv_x_Age = adv_x_Age * 86400.0 / self.volume
        adv_y_Age = adv_y_Age * 86400.0 / self.volume
        adv_r_Age = adv_r_Age * 86400.0 / self.volume
        #print(adv_x_Age.name)

        # sum up to get the total tendency due to advection
        advct_Age = (adv_x_Age + adv_y_Age + adv_r_Age).rename('advct_Age')
        #print(advct_Age.name)
        #print(adv_x_Age.name)
    
        adv_x_Age = adv_x_Age.rename('adv_x_Age')
        adv_y_Age = adv_y_Age.rename('adv_y_Age')
        adv_r_Age = adv_r_Age.rename('adv_r_Age')

        if self.terms is not None:
            self.terms['adv_x_Age'] = adv_x_Age
            self.terms['adv_y_Age'] = adv_y_Age
            self.terms['adv_r_Age'] = adv_r_Age
            self.terms['advct_Age'] = advct_Age
        else:
            self.terms = xr.merge([adv_x_Age, adv_y_Age, adv_r_Age, advct_Age])

    def cal_diffusion_tendency(self):
        '''
        Calculate tendency due to harmonic diffusion.
        
        Parameters
        ----------
        suffix : string
            A given string in ['TH', 'SLT'] for heat or salt calculation.
        '''
        # get MITgcm diagnostics
        DFxE = self.dset['DFxETr11']
        DFyE = self.dset['DFyETr11']
        DFrE = self.dset['DFrETr11']
        DFrI = self.dset['DFrITr11']

        # difference to get flux convergence, sign convention is opposite for vertical direction
        dffxE_Age = -self.grid.diff(DFxE, 'X', boundary='fill').rename('dffxE_Age')
        dffyE_Age = -self.grid.diff(DFyE, 'Y', boundary='fill').rename('dffyE_Age')
        dffrE_Age =  self.grid.diff(DFrE, 'Z', boundary='fill').rename('dffrE_Age')
        dffrI_Age =  self.grid.diff(DFrI, 'Z', boundary='fill').rename('dffrI_Age')

        # change unit to K/day
        dffxE_Age = dffxE_Age * 86400.0 / self.volume
        dffyE_Age = dffyE_Age * 86400.0 / self.volume
        dffrE_Age = dffrE_Age * 86400.0 / self.volume
        dffrI_Age = dffrI_Age * 86400.0 / self.volume

        # sum up to get the total tendency due to harmonic diffusion
        diffu_Age = (dffxE_Age + dffyE_Age + dffrE_Age + dffrI_Age).rename('diffu_Age')

        if self.terms is not None:
            self.terms['dffxE_Age'] = dffxE_Age
            self.terms['dffyE_Age'] = dffyE_Age
            self.terms['dffrE_Age'] = dffrE_Age
            self.terms['dffrI_Age'] = dffrI_Age
            self.terms['diffu_Age'] = diffu_Age
        else:
            self.terms = xr.merge([dffxE_Age, dffyE_Age, dffrE_Age, dffrI_Age, diffu_Age])


    def cal_true_tendency(self, yrs):
        '''
        Calculate true tendency output by the model.
        
        Parameters
        ----------
        yrs : float. Number of years in a between snapshots default is 1)
        '''
        # get MITgcm diagnostics
        AgeSn = self.snaps.TRAC11

        # calculate the true tendency
        Tend_Age = AgeSn.diff('time')/(yrs*360)
        # note that I've multiplied other terms to get into /day units. 
        # ooh I hope this doesn't mess it up! 
        Tend_Age = xr.where(Tend_Age ==0, np.nan, Tend_Age).rename('Tend_Age')
        #total_tdc = TOTTend.rename(f'total_{suffix[0]}tdc')

        if self.terms is not None:
            self.terms['Tend_Age'] = Tend_Age
        else:
            self.terms = xr.merge([Tend_Age])

    def cal_true_tendency(self):
        '''
        Calculate true tendency output by the model.
        
        Parameters
        ----------

        '''
        # get MITgcm diagnostics
        AgeSn = self.snaps.TRAC11

        # calculate the true tendency
        Tend_Age = AgeSn.diff('time')/(360)
        Tend_Age = Tend_Age.rename('Tend_Age')
        #total_tdc = TOTTend.rename(f'total_{suffix[0]}tdc')

        if self.terms is not None:
            self.terms['Tend_Age'] = Tend_Age
        else:
            self.terms = xr.merge([Tend_Age])

    def aging(self):

        Aging = xr.where(self.snaps.TRAC11==0, np.nan, 1/(360))
        Aging = Aging.rename('Aging')
        #total_tdc = TOTTend.rename(f'total_{suffix[0]}tdc')

        if self.terms is not None:
            self.terms['Aging'] = Aging
        else:
            self.terms = xr.merge([Aging])

## Now working on separating the divergent components. This is mission critical. 
# Need to bring in both the velocities and the TRAC11 average values.

    def cal_ND_adv(self): 

        # This removes the divergent components of the advection terms (for each dimension)
        # thus ND or non-divergent. 

        u_transport = self.dset.UVEL * self.coords.dyG * self.coords.hFacW * self.coords.drF
        v_transport = self.dset.VVEL * self.coords.dxG * self.coords.hFacS * self.coords.drF
        w_transport = self.dset.WVEL * self.coords.rA 

        div_u = self.grid.diff(u_transport, 'X')
        div_v = self.grid.diff(v_transport, 'Y') 
        div_w = self.grid.diff(w_transport, 'Z')
        #totdiv0 = div_u0 + div_v0 - div_w0

        if self.terms is None:
            self.cal_advection_tendency()

        advx_nd_Age = self.terms.adv_x_Age - (self.dset.TRAC11*div_u)*86400/self.volume
        advy_nd_Age = self.terms.adv_y_Age - (self.dset.TRAC11*div_v)*86400/self.volume
        advr_nd_Age = self.terms.adv_r_Age - (self.dset.TRAC11*div_w)*86400/self.volume

        # might not be necessary... 
        advx_nd_Age = advx_nd_Age.rename('advx_nd_Age')
        advy_nd_Age = advy_nd_Age.rename('advy_nd_Age')
        advr_nd_Age = advr_nd_Age.rename('advr_nd_Age')

        #if self.terms is not None:
        self.terms['advx_nd_Age'] = advx_nd_Age
        self.terms['advy_nd_Age'] = advy_nd_Age
        self.terms['advr_nd_Age'] = advr_nd_Age


    def calc_all(self):
        # just does everything so I can save time. 
        self.cal_true_tendency()
        self.cal_advection_tendency()
        self.cal_diffusion_tendency()
        self.cal_ND_adv()
        self.aging()

        resid_Age = self.terms.Tend_Age - self.terms.advct_Age - self.terms.diffu_Age - self.terms.Aging
        self.terms['resid_Age'] = resid_Age


# In[ ]:





# In[3]:


class OxygenBudget(object):
    '''
    This class is designed for the budget analysis in MITgcm.
    Credit to MiniUFO and Emily 
    '''
    def __init__(self, dset, snaps):
        '''
        Construct a Budget instance using a Dataset
        
        Parameters
        ----------
        dset : xarray.Dataset
            a given Dataset containing MITgcm output diagnostics
        
        Return
        ----------
        terms : xarray.Dataset
            A Dataset containing all budget terms
        '''
        self.grid   = xgcm.Grid(dset)
        self.coords = dset.coords.to_dataset().reset_coords()
        self.dset   = dset.reset_coords(drop=True)
        self.volume = dset.drF * dset.hFacC * dset.rA
        self.snaps  = snaps
        self.terms  = None

    ## it would be nice to just have all the terms calculated when we initialize, including the residual! 
    # maybe next coding project - should be very quick. 

    '''
    Calculate all the budget tendency terms.
    '''
    def cal_advection_tendency(self):
        '''
        Calculate tendency due to advection.
        
        Parameters
        ----------
        suffix : string
            A given string in ['TH', 'SLT'] for heat or salt calculation.
        '''
        # get MITgcm diagnostics
        ADVx = self.dset['ADVxTr05']
        ADVy = self.dset['ADVyTr05']
        ADVr = self.dset['ADVrTr05']

        # difference to get flux convergence, sign convention is opposite for verticl
        adv_x_O2 = -self.grid.diff(ADVx, 'X', boundary='fill').rename('adv_x_O2')
        adv_y_O2 = -self.grid.diff(ADVy, 'Y', boundary='fill').rename('adv_y_O2')
        adv_r_O2 =  self.grid.diff(ADVr, 'Z', boundary='fill').rename('adv_r_O2')

        #print(adv_x_O2.name)
        
        # change unit to K/day # somehow this step lost their names!
        adv_x_O2 = adv_x_O2 * 86400.0 / self.volume
        adv_y_O2 = adv_y_O2 * 86400.0 / self.volume
        adv_r_O2 = adv_r_O2 * 86400.0 / self.volume
        #print(adv_x_O2.name)

        # sum up to get the total tendency due to advection
        advct_O2 = (adv_x_O2 + adv_y_O2 + adv_r_O2).rename('advct_O2')
        #print(advct_O2.name)
        #print(adv_x_O2.name)
    
        adv_x_O2 = adv_x_O2.rename('adv_x_O2')
        adv_y_O2 = adv_y_O2.rename('adv_y_O2')
        adv_r_O2 = adv_r_O2.rename('adv_r_O2')

        if self.terms is not None:
            self.terms['adv_x_O2'] = adv_x_O2
            self.terms['adv_y_O2'] = adv_y_O2
            self.terms['adv_r_O2'] = adv_r_O2
            self.terms['advct_O2'] = advct_O2
        else:
            self.terms = xr.merge([adv_x_O2, adv_y_O2, adv_r_O2, advct_O2])

    def cal_diffusion_tendency(self):
        '''
        Calculate tendency due to harmonic diffusion.
        
        Parameters
        ----------
        suffix : string
            A given string in ['TH', 'SLT'] for heat or salt calculation.
        '''
        # get MITgcm diagnostics
        DFxE = self.dset['DFxETr05']
        DFyE = self.dset['DFyETr05']
        DFrE = self.dset['DFrETr05']
        DFrI = self.dset['DFrITr05']

        # difference to get flux convergence, sign convention is opposite for vertical direction
        dffxE_O2 = -self.grid.diff(DFxE, 'X', boundary='fill').rename('dffxE_O2')
        dffyE_O2 = -self.grid.diff(DFyE, 'Y', boundary='fill').rename('dffyE_O2')
        dffrE_O2 =  self.grid.diff(DFrE, 'Z', boundary='fill').rename('dffrE_O2')
        dffrI_O2 =  self.grid.diff(DFrI, 'Z', boundary='fill').rename('dffrI_O2')

        # change unit to K/day
        dffxE_O2 = dffxE_O2 * 86400.0 / self.volume
        dffyE_O2 = dffyE_O2 * 86400.0 / self.volume
        dffrE_O2 = dffrE_O2 * 86400.0 / self.volume
        dffrI_O2 = dffrI_O2 * 86400.0 / self.volume

        # sum up to get the total tendency due to harmonic diffusion
        diffu_O2 = (dffxE_O2 + dffyE_O2 + dffrE_O2 + dffrI_O2).rename('diffu_O2')

        if self.terms is not None:
            self.terms['dffxE_O2'] = dffxE_O2
            self.terms['dffyE_O2'] = dffyE_O2
            self.terms['dffrE_O2'] = dffrE_O2
            self.terms['dffrI_O2'] = dffrI_O2
            self.terms['diffu_O2'] = diffu_O2
        else:
            self.terms = xr.merge([dffxE_O2, dffyE_O2, dffrE_O2, dffrI_O2, diffu_O2])


    def cal_true_tendency(self, yrs):
        '''
        Calculate true tendency output by the model.
        
        Parameters
        ----------
        yrs : float. Number of years in a between snapshots default is 1)
        '''
        # get MITgcm diagnostics
        O2Sn = self.snaps.TRAC05

        # calculate the true tendency
        Tend_O2 = O2Sn.diff('time')/(yrs*360)
        # note that I've multiplied other terms to get into /day units. 
        Tend_O2 = Tend_O2.rename('Tend_O2')
        #total_tdc = TOTTend.rename(f'total_{suffix[0]}tdc')

        if self.terms is not None:
            self.terms['Tend_O2'] = Tend_O2
        else:
            self.terms = xr.merge([Tend_O2])

    def cal_true_tendency(self):
        '''
        Calculate true tendency output by the model.
        
        Parameters
        ----------

        '''
        # get MITgcm diagnostics
        O2Sn = self.snaps.TRAC05

        # calculate the true tendency
        Tend_O2 = O2Sn.diff('time')/(360)
        Tend_O2 = Tend_O2.rename('Tend_O2')
        #total_tdc = TOTTend.rename(f'total_{suffix[0]}tdc')

        if self.terms is not None:
            self.terms['Tend_O2'] = Tend_O2
        else:
            self.terms = xr.merge([Tend_O2])

    def cal_biological_SS(self):
        '''
        Calculate net O2 production minus remineralization
        
        Parameters
        ----------

        '''
        # get MITgcm diagnostics
        O2Bio = self.dset.UDIAG3*  86400.0

        # calculate the true tendency
        Bio_O2 =  O2Bio.rename('Bio_O2')
        #total_tdc = TOTTend.rename(f'total_{suffix[0]}tdc')

        if self.terms is not None:
            self.terms['Bio_O2'] = Bio_O2
        else:
            self.terms = xr.merge([Bio_O2])

    def cal_ND_adv(self): 

        # This removes the divergent components of the advection terms (for each dimension)
        # thus ND or non-divergent. 

        u_transport = self.dset.UVEL * self.coords.dyG * self.coords.hFacW * self.coords.drF
        v_transport = self.dset.VVEL * self.coords.dxG * self.coords.hFacS * self.coords.drF
        w_transport = self.dset.WVEL * self.coords.rA 

        div_u = self.grid.diff(u_transport, 'X')
        div_v = self.grid.diff(v_transport, 'Y') 
        div_w = self.grid.diff(w_transport, 'Z')
        #totdiv0 = div_u0 + div_v0 - div_w0

        if self.terms is None:
            self.cal_advection_tendency()

        advx_nd_O2 = self.terms.adv_x_O2 - (self.dset.TRAC05*div_u)*86400/self.volume
        advy_nd_O2 = self.terms.adv_y_O2 - (self.dset.TRAC05*div_v)*86400/self.volume
        advr_nd_O2 = self.terms.adv_r_O2 - (self.dset.TRAC05*div_w)*86400/self.volume

        # might not be necessary... 
        advx_nd_O2 = advx_nd_O2.rename('advx_nd_O2')
        advy_nd_O2 = advy_nd_O2.rename('advy_nd_O2')
        advr_nd_O2 = advr_nd_O2.rename('advr_nd_O2')

        #if self.terms is not None:
        self.terms['advx_nd_O2'] = advx_nd_O2
        self.terms['advy_nd_O2'] = advy_nd_O2
        self.terms['advr_nd_O2'] = advr_nd_O2


    def calc_all(self):
        # just does everything so I can save time. 
        self.cal_true_tendency()
        self.cal_advection_tendency()
        self.cal_diffusion_tendency()
        self.cal_ND_adv()
        self.cal_biological_SS()

        resid_O2 = self.terms.Tend_O2 - self.terms.advct_O2 - self.terms.diffu_O2 - self.terms.Bio_O2
        self.terms['resid_O2'] = resid_O2


# In[ ]:





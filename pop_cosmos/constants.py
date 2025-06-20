import numpy as np

## COSMOS_FLUX_SOFTENING ##
# flux softening parameters for the COSMOS bands
# units of nanomaggies (AB system)
# same order as COSMOS_FILTERS
COSMOS_FLUX_SOFTENING = np.array(
    [
        0.0010796025,
        0.0030850505,
        0.005081096,
        0.00087932684,
        0.0029128664,
        0.004059232,
        0.0026236728,
        0.0049892664,
        0.0033344517,
        0.0011464796,
        0.0052633565,
        0.003998706,
        0.006620309,
        0.0047184913,
        0.006163379,
        0.0011476493,
        0.0042965584,
        0.0062214015,
        0.0018217973,
        0.004118161,
        0.00823126,
        0.008909255,
        0.010833998,
        0.00792853,
        0.0032293492,
        0.0034518978,
    ]
)

## COSMOS_FILTERS ##
# list of COSMOS filter names in the order used by pop-cosmos
# names correspond to the Photulator emulators in the repo
COSMOS_FILTERS = [
    'u_megaprime_sagem',
    'hsc_g',
    'hsc_r',
    'hsc_i',
    'hsc_z',
    'hsc_y',
    'uvista_y_cosmos',
    'uvista_j_cosmos',
    'uvista_h_cosmos',
    'uvista_ks_cosmos',
    'ia427_cosmos',
    'ia464_cosmos',
    'ia484_cosmos',
    'ia505_cosmos',
    'ia527_cosmos',
    'ia574_cosmos',
    'ia624_cosmos',
    'ia679_cosmos',
    'ia709_cosmos',
    'ia738_cosmos',
    'ia767_cosmos',
    'ia827_cosmos',
    'NB711.SuprimeCam',
    'NB816.SuprimeCam',
    'irac1_cosmos',
    'irac2_cosmos'
]

COSMOS_FILTERS_LATEX = [
    '$u$',
    '$g$',
    '$r$',
    '$i$',
    '$z$',
    '$y$',
    '$Y$',
    '$J$',
    '$H$',
    '$K_S$',
    'IB427',
    'IB464',
    'IA484',
    'IB505',
    'IA527',
    'IB574',
    'IA624',
    'IA679',
    'IB709',
    'IA738',
    'IA767',
    'IB827',
    'NB711',
    'NB816',
    '$Ch.\\,1$',
    '$Ch.\\,2$'
]

## ZEROPOINTS_LEISTEDT23 ##
# flux zero point corrections for the COSMOS filters from Leistedt+23
# no units (fractional corrections)
ZEROPOINTS_LEISTEDT23 = np.array(
    [
        1.0068392 , 
        1.0830942 , 
        1.063653  , 
        1.        , 
        1.0044228 ,
        1.0459933 , 
        1.0172132 , 
        0.98858535, 
        0.96549976, 
        1.0593857 ,
        0.9618197 , 
        0.99145955, 
        1.0053478 , 
        1.0056049 , 
        0.9771116 ,
        0.93141526, 
        0.99394125, 
        1.1472832 , 
        0.96423334, 
        0.95098406,
        0.95391566, 
        0.9248251 , 
        0.98027456, 
        0.929084  , 
        0.9520671 ,
        0.92414236
    ], 
dtype=np.float32)

## ERRORFLOOR_LEISTEDT23 ##
# flux error floors for the COSMOS filters from Leistedt+23
# no units (fractional error floors)
ERRORFLOOR_LEISTEDT23 = np.array(
    [
        0.1       ,
        0.0022452 ,
        0.01825048,
        0.00614847,
        0.00856215,
        0.00819892,
        0.01924733,
        0.02077879,
        0.02989466,
        0.00777809,
        0.0437037 , 
        0.04945194, 
        0.02857511, 
        0.02843877, 
        0.03176803,
        0.03493299,
        0.0199778 , 
        0.02429752, 
        0.02285645, 
        0.01740251,
        0.03913081, 
        0.03608742, 
        0.02332224, 
        0.02462296, 
        0.1       ,
        0.1       
    ], 
dtype=np.float32)

## EMLINEOFFSETS_LEISTEDT23 ##
# emission line offsets from Leistedt+23
# no units (fractional correction)
EMLINEOFFSETS_LEISTEDT23 = np.array(
    [
        9.5367432e-07, 
       -9.0599060e-06,  
        3.9339066e-06,  
        0.0000000e+00,
        1.0728836e-06,  
        5.3644180e-06,  
        6.9141388e-06,  
        3.9339066e-06,
        0.0000000e+00, 
       -4.7683716e-07, 
       -6.7353249e-06, 
       -1.3113022e-06,
       -7.1525574e-07, 
       -5.6028366e-06, 
       -2.3841858e-07,  
        1.2457371e-03,
        2.5112629e-03, 
       -4.4107437e-06, 
       -6.4969063e-06, 
       -6.0200691e-06,
        1.5876293e-03, 
        1.3113022e-06,  
        6.1988831e-06, 
       -1.7881393e-06,
       -3.0398369e-06, 
       -6.0174286e-01, 
       -1.0000000e+00,  
        8.3446503e-06,
        1.9073486e-06,  
        4.7683716e-07,  
        4.8875809e-06,  
        3.2419527e-01,
       -6.4969063e-06, 
       -1.0000000e+00, 
       -1.4094931e-01,  
        2.5132895e-03,
       -5.1485682e-01,  
        2.6191330e-01, 
       -8.0184281e-02, 
       -3.3122957e-01,
       -2.1513700e-03, 
       -5.6559724e-01, 
       -3.4093344e-01,  
        1.1364901e-01
    ], 
dtype=np.float32)

## EMLINEERRORS_LEISTEDT23 ##
# emission line strength standard deviations from Leistedt+23
# no units (fractional)
EMLINEERRORS_LEISTEDT23 = np.array(
    [
        1.4252266e-13, 
        1.4242317e-13, 
        1.4635359e-13, 
        1.4334123e-13,
        1.4553604e-13, 
        1.4245789e-13, 
        1.4269427e-13, 
        1.4249889e-13,
        1.4250389e-13, 
        1.4258371e-13, 
        1.4256273e-13, 
        1.4248417e-13,
        1.4336201e-13, 
        1.4266451e-13, 
        1.4266074e-13, 
        2.1625620e-13,
        3.5995379e-13, 
        1.4243850e-13, 
        1.4367905e-13, 
        1.4603672e-13,
        1.0278946e-13, 
        1.4895650e-13, 
        1.4246227e-13, 
        1.4369894e-13,
        1.4251202e-13, 
        9.9999998e-14, 
        1.7024499e-13, 
        1.4645261e-13,
        1.4997000e-13, 
        1.4852341e-13, 
        1.4476311e-13, 
        1.5547955e-13,
        1.5363024e-13, 
        2.0169943e-13, 
        1.6893969e-13, 
        9.9999998e-14,
        1.9118050e-13, 
        9.9999998e-14, 
        1.8890240e-13, 
        1.0125776e-13,
        9.9999998e-14, 
        9.9999998e-14, 
        9.9999998e-14, 
        2.6402798e-02
    ], 
dtype=np.float32)
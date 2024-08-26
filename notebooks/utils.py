def fractional_uncertainty(value, delta):
    """Fractional uncertainty calculation:
    fractional_uncertainty = delta / value

    Parameters
    ----------
    value : float
        data point
    delta : float 
        uncertainty in the value

    Returns
    -------
    fractional uncertainty : float
        fractional uncertainty calculated as delta / value


    Examples
    --------
    frac_unc = fractional_uncertainty(10, 2) # returns 0.2
    """
    
    assert value != 0, f"!! value can not be 0"
    return delta / value

def fractional_uncertainty_squared(delta, value):
    """Fractional uncertainty squared calculation:
    frac_uncert_squared = (delta / value)**2

    Parameters
    ----------
    value : float
        data point
    delta : float 
        uncertainty in the value

    Returns
    -------
    fractional uncertainty_squared : float
        squared fractional uncertainty calculated as (delta / value)**2


    Examples
    --------
    frac_unc_squared = fractional_uncertainty_squared(10, 2) # returns 0.04
    """
    assert value != 0, f"!! value can not be 0"
    return fractional_uncertainty(delta, value) * fractional_uncertainty(delta, value)

def schmidt_number(temp_C):
    """Calculates the Schmidt number as defined by Jahne et al. (1987) and listed
    in Wanninkhof (2014) Table 1.

    Args:
        temp_C (array): temperature in degrees C

    Returns:
        array: Schmidt number (dimensionless)

    Examples:
        >>> schmidt_number(20)  # from Wanninkhof (2014)
        668.344

    References:
        Jähne, B., Heinz, G., & Dietrich, W. (1987). Measurement of the
        diffusion coefficients of sparingly soluble gases in water. Journal
        of Geophysical Research: Oceans, 92(C10), 10767–10776.
        https://doi.org/10.1029/JC092iC10p10767
    """
    from numpy import nanmedian

    if nanmedian(temp_C) > 270:
        raise ValueError("temperature is not in degC")

    T = temp_C

    a = +2116.8
    b = -136.25
    c = +4.7353
    d = -0.092307
    e = +0.0007555

    Sc = a + b * T + c * T ** 2 + d * T ** 3 + e * T ** 4

    return Sc

def d_schmidt_number_wrt_temp(temp_C):
    """Calculates the derivative of the Schmidt number 
    with respect to (wrt) temperature 

    Args:
        temp_C (array): temperature in degrees C

    Returns:
        array: derivative of Schmidt number wrt temperature, dSc/dT, (1/degC)

    Examples:
        >>> schmidt_number(20)  # from Wanninkhof (2014)
        668.344

    See Also:
        schmidt_number()
    """
    from numpy import nanmedian

    if nanmedian(temp_C) > 270:
        raise ValueError("temperature is not in degC")

    T = temp_C
    
    A = 2_116.8
    B = -136.25
    C = 4.7353
    D = -0.092307
    E = 0.000755
    return B + 2*C*temp_C + 3*D*(temp_C**2) + 4*E*(temp_C**3)

def weiss1974_f1(temp_C):
    T = temp_C + 273.15
    
    a1 = -58.0931
    a2 = +90.5069
    a3 = +22.2940
    T100 = T / 100
    return a1 + a2 * (100/T) + a3 * log(T100)

def weiss1974_f2(temp_C):
    T = temp_C + 273.15
    
    b1 = +0.027766
    b2 = -0.025888
    b3 = +0.0050578
    T100 = T / 100
    return b1 + b2 * T100 + b3 * T100 ** 2

def d_weiss1974_f1_wrt_temp(temp_C):
    T = temp_C + 273.15
    
    a1 = -58.0931
    a2 = +90.5069
    a3 = +22.2940
    T100 = T / 100
    return -a2 * 100 * (T ** (-2)) + a3 * (1/T)

def d_weiss1974_f2_wrt_temp(temp_C):
    T = temp_C + 273.15
    
    b1 = +0.027766
    b2 = -0.025888
    b3 = +0.0050578
    T100 = T / 100
    return (b2 /100) + ( (2 * b3 * T) / (100 ** 2) )

def d_weiss1974_wrt_temp(temp_C, S):
    """deriviative of solubility with respect to temperature
    parameterized using Weiss 1974
    """
    T = temp_C + 273.15
    
    df1_dt = d_weiss1974_f1_wrt_temp(T)
    df2_dt = d_weiss1974_f2_wrt_temp(T)
    
    return df1_dt + ( S * df2_dt )

def d_weiss1974_wrt_salinity(temp_C):
    """deriviative of solubility with respect to salinity
    parameterized using Weiss 1974
    """
    T = temp_C + 273.15
    d_ko_d_salt = weiss1974_f2(T)
    return d_ko_d_salt

def weiss1974(temp_C, S):
    T = temp_C + 273.15
    ko = weiss1974_f1(T) + ( S * weiss1974_f2(T) )
    return ko


def d_wann14_wrt_umean(temp_C, u_mean, a=0.251):
    """deriviative kw with respect to mean temp"""
    Sc = schmidt_number(temp_C)
    return a * (2 * u_mean) * (Sc / 660)**(0.5)

def d_wann14_wrt_ustd(temp_C, u_std, a=0.251):
    """deriviative kw with respect to standard deviation in temp"""
    Sc = schmidt_number(temp_C)
    return a * (2 * u_std) * (Sc / 660)**(0.5)

def d_wann14_wrt_ustd(temp_C, u_std, a=0.251):
    """deriviative kw with respect to standard deviation in temp"""
    Sc = schmidt_number(temp_C)
    return a * (2 * u_std) * (Sc / 660)**(0.5)

def frac_kw_umean(u_mean, u_std, delta_umean):
    numerator = 2 * u_mean * delta_umean 
    denominator = u_mean*u_mean + u_std*u_std
    return (numerator / denominator )

def frac_kw_ustd(u_mean, u_std, delta_ustd):
    numerator = 2 * u_std * delta_ustd
    denominator = u_mean*u_mean + u_std*u_std
    return (numerator / denominator )

def frac_kw_sc(temp_C, delta_T):
    Sc = schmidt_number(temp_C)
    dSc_dT = d_schmidt_number_wrt_temp(temp_C)
    delta_Sc = dSc_dT * delta_T
    return (0.5 * delta_Sc) / Sc

def frac_pco2ocn(pco2, delta_pco2):
    return delta_pco2 / pco2

def frac_ko_temp(temp_C, S, delta_T):
    T = temp_C + 273.15
    d_ko_dT = d_weiss1974_wrt_temp(T, S)
    return (d_ko_dT * delta_T) 

def frac_ko_salt(temp_C, delta_S):
    T = temp_C + 273.15
    d_ko_ds = d_weiss1974_wrt_salinity(T)
    return d_ko_ds * delta_S
import numpy as np

from pintle_pipeline.config_schemas import GraphiteInsertConfig

SIGMA = 5.670374419e-8
R = 8.314

activation_energy = graphite_insert.activation_energy
pressure_reference = graphite_insert.oxidation_reference_pressure
pressure_exponent = graphite_insert.oxidation_pressure_exponent
molar_mass_carbon = graphite_insert.molar_mass_carbon
k0 = graphite_insert.oxidation_pre_exponential
R_GAS = R
partial_pressure_oxygen = graphite_insert.partial_pressure_oxygen
pressure_reference = graphite_insert.oxidation_reference_pressure
pressure_exponent = graphite_insert.oxidation_pressure_exponent
activation_energy = graphite_insert.activation_energy


def oxidation_enthalpy_weighted_calc(oxidation_enthalpy_CO, oxidation_enthalpy_CO2, fraction_CO):
    return oxidation_enthalpy_CO * fraction_CO + oxidation_enthalpy_CO2 * (1 - fraction_CO)

def local_recession_rate_ox_calc(m_area_flux_ox, rho_s):
    return m_area_flux_ox / rho_s

def local_recession_rate_th_calc(m_area_flux_th, rho_s):
    return m_area_flux_th / rho_s

def local_recession_rate_total_calc(local_recession_rate_ox, local_recession_rate_th):
    return local_recession_rate_ox + local_recession_rate_th

#----------------------------------------------------------------------------------------------------------------------

def q_area_flux_in_calc(heat_transfer_coefficient, gas_temperature, surface_temperature, q_area_flux_rad_g_to_s):
    return h_g * (T_g - T_s) + q_area_flux_rad_g_to_s

def q_area_flux_rad_calc(esp, sigma, surface_temperature, environment_temperature):
    return esp * sigma * (surface_temperature**4 - environment_temperature**4)

def q_area_flux_feedback_calc(feedback_fraction, m_area_flux_ox, oxidation_enthalpy_weighted):
    return feedback_fraction * m_area_flux_ox * np.abs(oxidation_enthalpy_weighted)

#----------------------------------------------------------------------------------------------------------------------

def m_area_flux_ox_kin_calc(molar_mass_carbon = 0.012011, 
                            k0 = 1e10, 
                            activation_energy = , 
                            R_GAS, 
                            surface_temperature, 
                            partial_pressure_oxygen, 
                            pressure_reference, 
                            pressure_exponent):
    




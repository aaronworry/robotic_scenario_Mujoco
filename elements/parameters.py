G = 9.81

class Motor():
    def __init__(self, name):
        self.name = name
        self.Ra = 1.03                   # Armature resistance [Ω]
        self.La = 0.572e-3               # Armature inductance [H]
        self.Km = 33.5e-3                # Torque constant [N.m/A]
        self.ia_max = 15.0               # Stall current [A]
        self.omega_nl = 6710.0*(pi/30.0) # No load speed [rpm -> rad/s]

class Cube():
    def __init__(self):
        self.l = 0.15;                    # Structure side length [m]
        self.m_s = 0.40;                  # Structure mass [kg]
        self.b = 0.                       # Surface viscuous friction coefficient [N.m.s/rad]
        self.I_s_xx = 2.0e-3;             # Structure moment of inertia around x-y-z axis at center of mass [kg.m^2]
        self.I_c_xx = self.I_s_xx + self.m_s * self.l * self.l / 2. # Cube moment of inertia around x-y-z axis at pivot point [kg.m^2]
        self.I_c_xy = -self.m_w * self.l * self.l /4. # Cube product of inertia at pivot point [kg.m^2]
        
        
        """
        self.m_s_g_l = self.m_s * G * self.l
        self.omega_n_0 = sqrt(self.m_s_g_l*sqrt(3.0)/(self.I_c_xx-self.I_c_xy))
        self.omega_n_1 = self.b/(self.I_c_xx+2*self.I_c_xy)
        """
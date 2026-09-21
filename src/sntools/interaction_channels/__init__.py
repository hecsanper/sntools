from abc import ABC, abstractmethod

from sntools.event import Event


cherenkov_threshold = 0.77  # Cherenkov threshold of electron/positron in water
emass_threshold = 0.512 #Slightly above electron mass to avoid ZeroDivisionError in `es` channel.
# If a detector contains liquid scintillator, we modify this threshold in gen_evts() in channel.py


class BaseChannel(ABC):
    """Abstract base class that defines the interface for all submodules in `sntools.interaction_channels`."""
    def __init__(self, flv) -> None:
        self.flavor = flv

    def __repr__(self) -> str:
        return f"{self.__module__}.Channel('{self.flavor}')"

    @abstractmethod
    def generate_event(self, eNu, dirx, diry, dirz):
        """Return an event with the appropriate incoming/outgoing particles.

        Input:
            eNu: neutrino energy
            dirx, diry, dirz: direction of outgoing particle (normalized to 1)
        """
        eE = self.get_eE(eNu, dirz)

        # replace `0` with interaction channel code ±CZZZAAA, where
        # + is for neutrinos, - for antineutrinos
        # C is 1 for charged-current and 2 for neutral-current interaction
        # AAAZZZ depends on the nucleus, e.g. 006012 for carbon-12 or 008016 for oxygen-16
        evt = Event(0)

        # Replace numbers with the appropriate particle code listed at https://pdg.lbl.gov/current/mc-particle-id
        evt.incoming_particles.append([12, eNu, 0, 0, 1])  # incoming neutrino
        # evt.incoming_particles.append([11, mE, 0, 0, 1])  # electron at rest
        evt.outgoing_particles.append([11, eE, dirx, diry, dirz])  # outgoing electron
        return evt

    @abstractmethod
    def dSigma_dE(self, eNu, eE):
        """Return differential cross section in MeV^-2.

        Inputs:
            eNu: neutrino energy
            eE:  energy of outgoing (detected) particle
        """
        sigma = 0
        # if sigma is in cm^2, convert to MeV^-2: http://www.wolframalpha.com/input/?i=cm%2F(hbar+*+c)+in+MeV%5E(-1)
        # sigma *= (5.067731E10)**2
        return sigma

    @abstractmethod
    def dSigma_dCosT(eNu, cosT):
        """Return differential cross section in MeV^-2 as a function of the emission angle of the outgoing (detected) particle.

        Input:
            eNu:  neutrino energy (MeV)
            cosT: cosine of the angle between neutrino and outgoing (detected) particle
        """
        return 1.0

    @abstractmethod
    def get_eE(self, eNu, cosT):
        """Return energy (in MeV) of outgoing (detected) particle.

        Input:
            eNu:  neutrino energy (in MeV)
            cosT: cosine of the angle between neutrino and outgoing (detected) particle
        """
        return eNu

    @abstractmethod
    def bounds_eE(self, eNu, *args):
        """Return kinematic bounds for integration over eE.

        Input:
            eNu:  neutrino energy (in MeV)
            args: [ignore this]
        Output:
            list with minimum & maximum allowed energy of outgoing (detected) particle
        """
        return (0, eNu)
    
    # Centralized upper neutrino energy limit.
    E_NU_MAX = 125.0 
    
    @property
    def bound_eNu_min(self):
        raise NotImplementedError(
            f"Critical error: The channel '{self.__class__.__name__}' has no defined minimum neutrino energy. "
            "You must define 'eNu_min' explicitly in the channel's file. "
            "If the channel has no physical threshold (e.g., ES), set 'eNu_min = 0.0'."
        )    

    @property
    def bounds_eNu(self):
        """Tuple with minimum & maximum energy of incoming neutrino."""
        return (self.bound_eNu_min, self.E_NU_MAX)

    def _bounds_eNu(self, eE):
        """Min/max neutrino energy that can produce a given positron energy.

        Optional. Can reduce numerical inaccuracy when integrating over eNu.
        """
        return self.bounds_eNu

    def _opts(self, eNu, *args):
        """Options for numerical integration with `scipy.nquad`.

        Optional. Return a list of values of eE where dSigma_dE(eNu, eE) has a discontinuity, to reduce numerical inaccuracy.
        """
        return {'points': []}

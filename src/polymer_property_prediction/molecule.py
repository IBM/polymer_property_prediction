class Molecule:
    def __init__(self,
                 name='',
                 smiles='',
                 temperature=0,
                 pressure=0,
                 polymer_concentration_wt=0,
                 Mn=0):
        self.name = name
        self.smiles = smiles
        self.temperature = temperature
        self.pressure = pressure
        self.polymer_concentration_wt = polymer_concentration_wt
        self.Mn = Mn

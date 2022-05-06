#######################################################################
#
#   This program calculates a set of Homopolymer properties
#
#   This program was implemented based on the Jocef Bicerano book:
#   Prediction of Polymer Properties
#   Third Edition
#   Marcel Dekker Inc., New York (2002)
#
#   INPUT: polymer-input.csv
#          polymer name; OPSIN monomer SMILES code
#          Ghost atom connected to the head atom represented as [*:1]
#          Ghost atom connected to the tail atom represented as [*:2]
#          OPSIN: Open Parser for Systematic IUPAC nomenclature:
#          https://opsin.ch.cam.ac.uk/
#
#
#   OUTPUT: polymer-output.csv
#
#   This python tool was written by:
#   Ronaldo Giro  10, May 2019
#   IBM Research - Brazil, Rio Lab
#
#######################################################################
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import math
import rdkit.Chem.Descriptors as rdcd
import rdkit.Chem.rdMolDescriptors as rdcmd
import pandas as pd

#######################################################################
#
#   Functions Section
#
#######################################################################


def calculateMol(smilesMol):
    # defining and initializing of some variables
    v_connec_index = np.zeros(4, dtype=np.float64)
    v_mechprop = np.zeros(5, dtype=np.float64)
    v_viscosity = np.zeros(2, dtype=np.float64)
    v_crosslink_index = np.zeros(2, dtype=np.int32)
    atom_indexes = np.zeros(2, dtype=np.int32)
    molar_volume = 0   # volume at room temperature 298 K
    density_monomer = 0
    Ecoh1 = 0
    Nmv = 0
    Natomic = 0
    Ngroup = 0
    Nfedors = 0
    Ntg = 0
    Nper = 0
    # water solubility parameter - from Hansen Solubility Parameters A Users Handbook 2nd Edition,
    # Charles M. Hansen
    Solb_w = math.sqrt((15.1)**2 + (20.4)**2 + (16.5)**2)

    lin = [smilesMol.name, smilesMol.smiles]
    smiles_opsin = lin[1]
    smiles_mol = ConvertOpsinToMolSmiles(smiles_opsin)
    smiles_mol = smiles_mol.replace('()', '')
    mol = Chem.MolFromSmiles(smiles_mol)
    atom_indexes = HeadTailAtoms(smiles_opsin)
    lin.append(str(atom_indexes[0]))
    lin.append(str(atom_indexes[1]))
    try:
        checkValenceProblems(mol, lin)
    except Exception as e:
        raise ValueError(str(e))
    v_connec_index = ConnectivityIndex(mol, lin)
    x0 = v_connec_index[0]
    x0v = v_connec_index[1]
    x1 = v_connec_index[2]
    x1v = v_connec_index[3]
    Nmv = NmvFunction(mol, lin)
    molar_volume = 3.642770 * x0 + 9.798697 * x0v - \
        8.542819 * x1 + 21.693912 * x1v + 0.978655 * Nmv
    # Subtracted two hydrogens due to head and tail periodic monomeric bond
    molar_mass = rdcd.ExactMolWt(mol) - 2.016
    density_monomer = molar_mass / molar_volume  # in g/cm^3
    Natomic = NatomicFunction(mol, lin)
    Ngroup = NgroupFunction(mol, lin)
    Nfedors = 6 * Natomic + 5 * Ngroup
    Ecoh1 = 9882.5 * x1 + 358.7 * Nfedors  # Cohesive Energy in J/mole
    Solb_p = math.sqrt(Ecoh1 /
                       molar_volume)  # polymer solubility in (J/cm^3)**0.5
    # As it approaches 1 means that it is more soluble in water
    solb_ratio = Solb_p / Solb_w
    x1_Ntg = x1NtgFunction(mol, lin)
    x2_Ntg = x2NtgFunction(mol, lin)
    x3_Ntg = x3NtgFunction(mol, lin)
    x4_Ntg = x4NtgFunction(mol, lin)
    x5_Ntg = x5NtgFunction(mol, lin)
    x6_Ntg = x6NtgFunction(mol, lin)
    x7_Ntg = x7NtgFunction(mol, lin)
    x8_Ntg = x8NtgFunction(mol, lin)
    x9_Ntg = x9NtgFunction(mol, lin)
    x10_Ntg = x10NtgFunction(mol, lin)
    x11_Ntg = x11NtgFunction(mol, lin)
    x12_Ntg = x12NtgFunction(mol, lin)
    x13_Ntg = x13NtgFunction(mol, lin)
    Ntg = 15 * x1_Ntg - 4 * x2_Ntg + 23 * x3_Ntg + 12 * x4_Ntg - 8 * x5_Ntg - 4 * x6_Ntg \
        - 8 * x7_Ntg + 5 * x8_Ntg + 11 * x9_Ntg + \
        8 * x10_Ntg - 11 * x11_Ntg - 4 * x12_Ntg
    natom = mol.GetNumAtoms()
    # Glass Transition temperature in Kelvin
    Tg = 351.00 + 5.63 * Solb_p + 31.68 * (Ntg / natom) - 23.94 * x13_Ntg
    molar_volume_temperature = molvtempFunction(mol, molar_volume, Tg,
                                                smilesMol.temperature)
    density_monomer_T = molar_mass / molar_volume_temperature  # in g/cm^3
    fh_parameter = FloryHugginsFunction(
        mol, Solb_w, Solb_p, smilesMol.temperature)
    density_water = waterdensityFunction(smilesMol.temperature,
                                         smilesMol.pressure) * 0.001  # in g/cm^3
    water_viscosity = waterviscosityFunction(
        smilesMol.temperature, smilesMol.pressure)
    v_viscosity = solutionviscosityFunction(mol, smilesMol.polymer_concentration_wt,
                                            fh_parameter, smilesMol.Mn, molar_mass,
                                            water_viscosity, lin, Solb_w,
                                            Solb_p, density_monomer_T)
    intrinsic_viscosity = v_viscosity[0]  # in cm^3/g
    solution_viscosity = v_viscosity[1]  # in Pa.s
    temp_half_decomposition = temperature_half_decomposition_Function(
        mol, lin, molar_mass)
    v_mechprop = mechanical_properties_Function(mol, lin, smilesMol.temperature, Tg,
                                                Ecoh1, molar_volume)
    '''
    # Never Used
    backbonetuple = BackboneFunction(mol, lin)
    sidegrouptuple = SideGroupFunction(mol, lin)
    ringstuple = RingsFunction(mol)
    rigidring = RigidRingFunction(mol)
    backboneringtuple = BackboneRingFunction(mol, lin)
    fusedgrouptupleindex = GroupFusedRingsFunction(mol)
    '''
    v_crosslink_index = physical_crosslink_Function(mol)
    # according to eq. (3.9) Josef Bicerano's book - simplest one implementation
    vdw_molar_volume = 3.861803 * x0 + 13.748435 * x1v
    N_BBrot = x5_Ntg
    N_SGrot = x6_Ntg
    N_rot = N_BBrot + N_SGrot
    Nper = NperFunction(mol, lin)
    newchor = Ecoh1 / molar_volume - 196 * \
        (molar_volume / vdw_molar_volume) + 110 * \
        (N_rot / natom) - 57 * (Nper / natom)  # eq. 15.15
    # eq. (15.17) Josef Bicerano's book in Dow Units -> (cm^3*(0.001 inches))
    # /(day*(100 inches^2)*atm)
    Permeability_O2 = 4991.6 * math.exp(-0.017622 * newchor)
    # eq. (15.18) Josef Bicerano's book in Dow Units
    Permeability_N2 = 1692 * math.exp(-0.019038 * newchor)
    # eq. (15.19) Josef Bicerano's book in Dow Units
    Permeability_CO2 = 31077 * math.exp(-0.019195 * newchor)
    #
    #  Permeability unit most used in literature -> barrer = 1x10^-10 cm^3(STP)*cm/(cm^2*s*cmHg) in
    # non-SI units in SI units 1 barrer = 3.35x10^-16 mol*m/(m^2*s*Pa)
    #
    DU_to_barrer = ((0.001 * 2.54) /
                    (24 * 3600 * 100 * 2.54 * 2.54 * 76)) / (1.0e-10)

    # WRITE POLIMER
    d = {'polymer_name': [lin[0]],
         'opsin_smiles': [smiles_opsin],
         'head_atom': [lin[2]],
         'tail_atom': [lin[3]],
         'smiles': [smiles_mol],
         't_k': [str(smilesMol.temperature)],
         'p_pa': [str(smilesMol.pressure)],
         'concentration_wt': [str(smilesMol.polymer_concentration_wt)],
         'mw_gmol': [str(smilesMol.Mn)],
         'density_at_298_k_gcm3': [str(density_monomer)],
         'density_at_t_gcm3': [str(density_monomer_T)],
         'molar_volume_cm3mol': [str(molar_volume)],
         'ecoh1_at_298_k_jmol': [str(Ecoh1)],
         'solub_ratio': [str(solb_ratio)],
         'fh_parameter': [str(fh_parameter)],
         'tg_k': [str(Tg)],
         'water_density_at_p_t_gcm3': [str(density_water)],
         'water_viscosity_at_t_pas': [str(water_viscosity)],
         'solution_viscosity_at_t_pas': [str(solution_viscosity)],
         'intrinsic_viscosity_cm3g': [str(intrinsic_viscosity)],
         'temperature_of_half_decomposition_k': [str(temp_half_decomposition)],
         'bulk_modulus_mpa': [str(v_mechprop[0])],
         'youngs_modulus_mpa': [str(v_mechprop[1])],
         'shear_modulus_mpa': [str(v_mechprop[2])],
         'brittle_fracture_stress_mpa': [str(v_mechprop[3])],
         'tensile_yield_stress_mpa': [str(v_mechprop[4])],
         'number_hydrogen_bonding': [str(v_crosslink_index[0])],
         'charge_of_counterion': [str(v_crosslink_index[1])],
         'permeability_co2_barrer': [str(Permeability_CO2 * DU_to_barrer)],
         'permeability_n2_barrer': [str(Permeability_N2 * DU_to_barrer)],
         'permeability_o2_barrer': [str(Permeability_O2 * DU_to_barrer)],
         'selectivity_co2_n2': [str(Permeability_CO2 / Permeability_N2)],
         'selectivity_o2_n2': [str(Permeability_O2 / Permeability_N2)]
         }

    ppf_df = pd.DataFrame(data=d)
    return ppf_df


def checkValenceProblems(mol, lin):
    # This function add the head and tail atoms of a molecule to a new
    # molecule in order to take into account the polymer periodicity
    # effect. So, molpf is mol with two added atoms: the head and the
    # tail atoms from mol
    #
    # smiles_string is the OPSIN: Open Parser for Systematic IUPAC
    # nomenclature SMILES with the periodic markers: [*:1] and [*:2]
    # for head and tail ghost atoms
    #
    # defining the molecule with periodic bonds to calculate fragments
    bi = int(lin[2])
    bf = int(lin[3])
    atomi = mol.GetAtomWithIdx(bi)
    atomf = mol.GetAtomWithIdx(bf)
    sti = atomi.GetSymbol()
    stf = atomf.GetSymbol()
    if sti == 'Si':
        sti = '[Si]'
    if stf == 'Si':
        stf = '[Si]'

    # replacing the head and tail ghost atoms [*:1] and [*:2] with the sti and stf atoms
    sttmp1 = lin[1]
    sttmp2 = sttmp1.replace("[*:1]", stf)
    st = sttmp2.replace("[*:2]", sti)
    molpf = Chem.MolFromSmiles(st, sanitize=False)
    molpf.UpdatePropertyCache()
    return molpf


def NmvFunction(mol, lin):
    # ---------------------------------------------------------------------
    #
    # 	    Nmv is a group contribution correction term for Molar Volume
    #
    # ---------------------------------------------------------------------
    deltaf = DeltaFunction(mol, lin)
    deltavf = DeltavFunction(mol, lin)
    Nsi = 0
    Ns = 0
    Nsulfone = 0
    Ncl = 0
    Nbr = 0
    Nester = 0
    Nether = 0
    Ncarbonate = 0
    Ncdouble = 0
    Ncyc = 0
    Nfused = 0
    # calculating the number of Si (Silicon) atoms
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if (z == 14):  # atomic number of Si (Silicon)
            Nsi = Nsi + 1

    # calculating the number of -S- (Sulfor) atoms
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        i = atom.GetIdx()
        if (z == 16):
            if (deltaf[i] == 1 and deltavf[i] == 5 / 9):
                Ns = Ns + 1
            if (deltaf[i] == 2 and deltavf[i] == 2 / 3):
                Ns = Ns + 1

    # calculating the number of Sulfone groups
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        i = atom.GetIdx()
        if (z == 16):
            if (deltaf[i] == 4 and deltavf[i] == 8 / 3):
                Nsulfone = Nsulfone + 1

    # calculating the number of Cl (Chlorine) atoms
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if (z == 17):  # atomic number of Cl (Chlorine)
            Ncl = Ncl + 1

    # calculating the number of Br (Bromine) atoms
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if (z == 35):  # atomic number of Br (Bromine)
            Nbr = Nbr + 1

    # defining the molecule with periodic bonds to calculate fragments
    molp = molextFunction(mol, lin)

    # calculating the number of backbone esters (R-C(=O)OR) groups
    Nester = Chem.Fragments.fr_ester(molp)

    # calculating the number of ether (R-O-R) groups
    Nether = Chem.Fragments.fr_ether(molp)

    # calculating the number of carbonate (-OCOO-) groups
    Ncarbonate = Chem.Fragments.fr_COO2(molp)

    nbond = mol.GetNumBonds()
    for i in range(0, nbond):
        bi = mol.GetBondWithIdx(i).GetBeginAtomIdx()
        atomi = mol.GetAtomWithIdx(bi)
        zi = atomi.GetAtomicNum()
        bf = mol.GetBondWithIdx(i).GetEndAtomIdx()
        atomf = mol.GetAtomWithIdx(bf)
        zf = atomf.GetAtomicNum()
        nbondtype = mol.GetBondWithIdx(i).GetBondTypeAsDouble()
        if (zi == 6 and zf == 6):
            if nbondtype == 2:
                if not atomi.IsInRing():
                    if not atomf.IsInRing():
                        Ncdouble = Ncdouble + 1

    # calculating the number of saturated rings Ncyc
    Ncyc = rdcmd.CalcNumSaturatedRings(mol)

    # calculating the number of fused ring Nfuse
    fusedtupleindex = FusedRingsFunction(mol)
    Nfused = len(fusedtupleindex)

    # calculating Nmv
    if Nfused >= 2:
        Nmvf = 24 * Nsi - 18 * Ns - 5 * Nsulfone - 7 * Ncl - 16 * Nbr
        + 2 * Nester + 3 * Nether + 5 * Ncarbonate + 5 * Ncdouble - 11 * Ncyc - 7 * (Nfused - 1)
    else:
        Nmvf = 24 * Nsi - 18 * Ns - 5 * Nsulfone - 7 * Ncl - 16 * Nbr + 2 * \
            Nester + 3 * Nether + 5 * Ncarbonate + 5 * Ncdouble - 11 * Ncyc

    return Nmvf


def NatomicFunction(mol, lin):
    # ---------------------------------------------------------------------
    #
    # 	    Natomic is an atomic correction term for Cohesive Energy
    #
    # ---------------------------------------------------------------------
    Ns = 0
    Nsulfone = 0
    Nf = 0
    Ncl = 0
    Nbr = 0
    Ncyanide = 0
    Natomicf = 0
    deltaf = DeltaFunction(mol, lin)
    deltavf = DeltavFunction(mol, lin)
    # calculating the number of -S- (Sulfor) atoms
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        i = atom.GetIdx()
        if (z == 16):
            if (deltaf[i] == 1 and deltavf[i] == 5 / 9):
                Ns = Ns + 1
            if (deltaf[i] == 2 and deltavf[i] == 2 / 3):
                Ns = Ns + 1

    # calculating the number of Sulfone groups
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        i = atom.GetIdx()
        if (z == 16):
            if (deltaf[i] == 4 and deltavf[i] == 8 / 3):
                Nsulfone = Nsulfone + 1

    # calculating the number of F (Fluorine) atoms
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if (z == 9):  # atomic number of Br (Fluorine)
            Nf = Nf + 1

    # calculating the number of Cl (Chlorine) atoms
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if (z == 17):  # atomic number of Cl (Chlorine)
            Ncl = Ncl + 1

    # calculating the number of Br (Bromine) atoms
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if (z == 35):  # atomic number of Br (Bromine)
            Nbr = Nbr + 1

    # calculating the number of cyanide group atoms
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        i = atom.GetIdx()
        if (z == 7):
            if (deltaf[i] == 1 and deltavf[i] == 5):
                Ncyanide = Ncyanide + 1

    # calculating Natomic
    Natomicf = 4 * Ns + 12 * Nsulfone - Nf + 3 * Ncl + 5 * Nbr + 7 * Ncyanide

    return Natomicf


def NgroupFunction(mol, lin):
    # ---------------------------------------------------------------------
    #
    # 	    Ngroup is an group correction term for Cohesive Energy
    #
    # ---------------------------------------------------------------------
    Nhydroxyl = 0
    Namide = 0
    Nnon_amide_nh = 0
    Nalkyl_ether = 0
    Ncdouble = 0
    Ncdouble_o_n = 0
    Ncdouble_o = 0
    Ncdouble_o_others = 0
    Narom_n = 0

    # defining the molecule with periodic bonds to calculate fragments
    bi = int(lin[2])
    bf = int(lin[3])
    atomi = mol.GetAtomWithIdx(bi)
    atomf = mol.GetAtomWithIdx(bf)
    sti = atomi.GetSymbol()
    stf = atomf.GetSymbol()

    molp = molextFunction(mol, lin)

    # calculating the number of hydroxyl in alcohol or phenol
    natomp = molp.GetNumAtoms()
    Noh_alcohol = 0
    Noh_phenol = 0
    Noh_alcohol = Chem.Fragments.fr_Al_OH(molp)
    Noh_phenol = Chem.Fragments.fr_Ar_OH(molp)
    if (sti == 'O' or stf == 'O'):
        if Noh_alcohol > 0:
            Noh_alcohol = Noh_alcohol - 1
    if (sti == 'O' or stf == 'O'):
        if (molp.GetAtomWithIdx(1).IsInRing()
                or molp.GetAtomWithIdx(natomp - 2).IsInRing()):
            Noh_phenol = Noh_phenol - 1
    Nhydroxyl = Noh_alcohol + Noh_phenol

    # calculating the number of non amide NH units
    patt = Chem.MolFromSmarts('C(=O)N')
    rm = AllChem.DeleteSubstructs(molp, patt)
    Nnon_amide_nh = Chem.Fragments.fr_NH1(rm)

    # calculating the number of alkyl ether
    patt1 = Chem.MolFromSmarts('c1ccccc1')
    rm1 = AllChem.DeleteSubstructs(molp, patt1)
    patt2 = Chem.MolFromSmarts('CC(=O)O')
    rm2 = AllChem.DeleteSubstructs(rm1, patt2)
    Nalkyl_ether = Chem.Fragments.fr_ether(rm2)

    # calculating the number of non-amide C=O next to Nitrogen (tertiary amide)
    patt1 = Chem.MolFromSmarts('C(=O)N(c)C')
    patt2 = Chem.MolFromSmarts('C(=O)N(C)C')
    n1 = len(molp.GetSubstructMatches(patt1))
    n2 = len(molp.GetSubstructMatches(patt2))
    Ncdouble_o_n = n1 + n2

    # calculating the number of amide groups
    Namide = Chem.Fragments.fr_amide(molp) - Ncdouble_o_n

    # calculating the number of C=O in carboxylic acid (-COOH), ketone (R(C=O)R') and aldehyde
    # R(C=O)H), R and R' != H
    Ncdouble_o = Chem.Fragments.fr_COO(molp) + Chem.Fragments.fr_ketone(
        molp) + Chem.Fragments.fr_aldehyde(molp)

    # calculating the number of C=O others
    Ncdouble_o_others = Chem.Fragments.fr_C_O(
        molp) - Ncdouble_o - Chem.Fragments.fr_amide(molp)

    # calculating the number of Nitrogen in aromatic six-membered ring
    patt = Chem.MolFromSmarts('c1ccncc1')
    Narom_n = len(molp.GetSubstructMatches(patt))

    # calculating the number of C=C bonds
    nbond = mol.GetNumBonds()
    for i in range(0, nbond):
        bi = mol.GetBondWithIdx(i).GetBeginAtomIdx()
        atomi = mol.GetAtomWithIdx(bi)
        zi = atomi.GetAtomicNum()
        bf = mol.GetBondWithIdx(i).GetEndAtomIdx()
        atomf = mol.GetAtomWithIdx(bf)
        zf = atomf.GetAtomicNum()
        nbondtype = mol.GetBondWithIdx(i).GetBondTypeAsDouble()
        if (zi == 6 and zf == 6):
            if nbondtype == 2:
                if not atomi.IsInRing():
                    if not atomf.IsInRing():
                        Ncdouble = Ncdouble + 1

    # calculating Ngroup
    Ngroupf = 12 * Nhydroxyl + 12 * Namide + 2 * Nnon_amide_nh - Nalkyl_ether - \
        Ncdouble + 4 * Ncdouble_o_n + 7 * Ncdouble_o + \
        2 * Ncdouble_o_others + 4 * Narom_n

    return Ngroupf


def NperFunction(mol, lin):
    # ---------------------------------------------------------------------
    #
    # 	    Nper is an group correction term for gas Permeability through
    #    polymer membrane
    #
    #    see eq. 15.14 from Josef Bicerano's book
    #
    # ---------------------------------------------------------------------
    molp = molextFunction(mol, lin)
    deltaf = DeltaFunction(mol, lin)
    deltavf = DeltavFunction(mol, lin)
    atom_indexesf2 = np.zeros(2, dtype=np.int32)
    atom_indexesf2 = HeadTailAtomsMolext(mol, lin)
    beginatomp = int(atom_indexesf2[0])
    endatomp = int(atom_indexesf2[1])
    linf = []
    linf.append(' ')
    linf.append(' ')
    linf.append(str(beginatomp))
    linf.append(str(endatomp))
    backbonetuplep = BackboneFunction(molp, linf)

    # calculating the number of C=C bonds
    Ncdouble = 0
    nbond = mol.GetNumBonds()
    for i in range(0, nbond):
        bi = mol.GetBondWithIdx(i).GetBeginAtomIdx()
        atomi = mol.GetAtomWithIdx(bi)
        zi = atomi.GetAtomicNum()
        bf = mol.GetBondWithIdx(i).GetEndAtomIdx()
        atomf = mol.GetAtomWithIdx(bf)
        zf = atomf.GetAtomicNum()
        nbondtype = mol.GetBondWithIdx(i).GetBondTypeAsDouble()
        if (zi == 6 and zf == 6):
            if nbondtype == 2:
                if not atomi.IsInRing():
                    if not atomf.IsInRing():
                        Ncdouble = Ncdouble + 1

    # calculating the number of ester (-COO-) groups in the backbone of the repeat unit
    Nbbester = 0
    patt_ester = Chem.MolFromSmarts('[#6][CX3](=O)[OX2H0][#6]')
    patt_tuple = molp.GetSubstructMatches(patt_ester)
    for i in range(0, len(patt_tuple)):
        tmpvector1 = np.intersect1d(patt_tuple[i], backbonetuplep)
        if len(tmpvector1) >= 3:
            Nbbester = Nbbester + 1

    # calculating x4_Nper
    x4_Nper = 0
    rigidring = RigidRingFunction(mol)
    ringstuple = RingsFunction(mol)
    fusedtupleindex = FusedRingsFunction(mol)
    backboneringtuple = BackboneRingFunction(mol, lin)

    # Finding the shortest path
    beginatom = int(lin[2])
    endatom = int(lin[3])
    shortestpathtuple = Chem.rdmolops.GetShortestPath(mol, beginatom, endatom)

    # Finding the bonds between the atoms of molecule
    bondidxs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                for b in mol.GetBonds()]
    bondidxs.append((beginatom, endatom))

    tmpvector = np.zeros([100], dtype=np.int64)
    for i in range(0, len(rigidring)):
        pp = rigidring[i]
        test = np.isin(pp, backboneringtuple)
        if test:
            tmpvector = ringstuple[pp]
            sideatomidxtmp = []
            sideatomidx = []
            for j in range(0, len(
                    bondidxs)):  # looking for side atoms attached to the ring
                test1 = np.isin(tmpvector, bondidxs[j][0])
                test2 = np.isin(tmpvector, bondidxs[j][1])
                if (np.any(test1) or np.any(test2)):
                    if not np.any(test1):
                        sideatomidxtmp.append(bondidxs[j][0])
                    if not np.any(test2):
                        sideatomidxtmp.append(bondidxs[j][1])
            sideatomidx = np.setdiff1d(
                sideatomidxtmp,
                shortestpathtuple)  # side atoms attached to the ring i
            x4_Nper = x4_Nper + len(sideatomidx)

    tmpvector = np.zeros([100], dtype=np.int64)
    sideatomidx = []
    for i in range(0, len(fusedtupleindex)):
        pp = fusedtupleindex[i]
        test = np.isin(pp, backboneringtuple)
        if test:
            tmpvector = ringstuple[pp]
            sideatomidxtmp1 = []
            sideatomidxtmp2 = []
            sideatomidx = []
            for j in range(0, len(
                    bondidxs)):  # looking for side atoms attached to the ring
                test1 = np.isin(tmpvector, bondidxs[j][0])
                test2 = np.isin(tmpvector, bondidxs[j][1])
                if (np.any(test1) or np.any(test2)):
                    if not np.any(test1):
                        sideatomidxtmp1.append(bondidxs[j][0])
                    if not np.any(test2):
                        sideatomidxtmp1.append(bondidxs[j][1])
            sideatomidxtmp2 = np.setdiff1d(sideatomidxtmp1, shortestpathtuple)
            for j in range(0, len(fusedtupleindex)):
                sideatomidx = np.setdiff1d(sideatomidxtmp2, ringstuple[
                    fusedtupleindex[j]])  # side atoms attached to the ring i
                sideatomidxtmp2 = sideatomidx
            x4_Nper = x4_Nper + len(sideatomidx)

    # calculating the number of Cl (Chlorine) atoms
    Ncl = 0
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if (z == 17):  # atomic number of Cl (Chlorine)
            ii = atom.GetIdx()  # atom index of Cl (Chlorine)
            for i in range(0, len(bondidxs)):  # find the atom bonded to Cl
                if (bondidxs[i][0] == ii):
                    jj = bondidxs[i][
                        1]  # index of atom bonded to Cl (Chlorine)
                if (bondidxs[i][1] == ii):
                    jj = bondidxs[i][
                        0]  # index of atom bonded to Cl (Chlorine)
            atom_n = mol.GetAtomWithIdx(jj)
            z_n = atom_n.GetAtomicNum()
            if (z_n == 6):
                if (deltaf[jj] == deltavf[jj]):
                    tmpvector1 = []
                    for i in range(
                            0, len(bondidxs)
                    ):  # find the atoms bonded to the carbon atom which is bonded to Cl
                        if (bondidxs[i][0] == jj or bondidxs[i][1] == jj):
                            tmpvector1.append(bondidxs[i])
                    tmpvector2 = []
                    for i in range(0, len(tmpvector1)):
                        if (tmpvector1[i][0] != beginatom
                                and tmpvector1[i][1] != endatom):
                            bondtmp = mol.GetBondBetweenAtoms(
                                tmpvector1[i][0], tmpvector1[i][1])
                            nbondtype = bondtmp.GetBondTypeAsDouble()
                            if nbondtype == 1:  # it is a single bond
                                tmpvector2.append(1)
                    test1 = np.isin(tmpvector2, 1)
                    if np.all(test1):
                        Ncl = Ncl + 1

    # calculating the number of Br (Bromine) atoms
    Nbr = 0
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if (z == 35):  # atomic number of Br (Bromine)
            ii = atom.GetIdx()  # atom index of Br (Bromine)
            for i in range(0, len(bondidxs)):  # find the atom bonded to Br
                if (bondidxs[i][0] == ii):
                    jj = bondidxs[i][1]  # index of atom bonded to Br (Bromine)
                if (bondidxs[i][1] == ii):
                    jj = bondidxs[i][0]  # index of atom bonded to Br (Bromine)
            atom_n = mol.GetAtomWithIdx(jj)
            z_n = atom_n.GetAtomicNum()
            if (z_n == 6):
                if (deltaf[jj] == deltavf[jj]):
                    tmpvector1 = []
                    for i in range(
                            0, len(bondidxs)
                    ):  # find the atoms bonded to the carbon atom which is bonded to Br
                        if (bondidxs[i][0] == jj or bondidxs[i][1] == jj):
                            tmpvector1.append(bondidxs[i])
                    tmpvector2 = []
                    for i in range(0, len(tmpvector1)):
                        if (tmpvector1[i][0] != beginatom
                                and tmpvector1[i][1] != endatom):
                            bondtmp = mol.GetBondBetweenAtoms(
                                tmpvector1[i][0], tmpvector1[i][1])
                            nbondtype = bondtmp.GetBondTypeAsDouble()
                            if nbondtype == 1:  # it is a single bond
                                tmpvector2.append(1)
                    test1 = np.isin(tmpvector2, 1)
                    if np.all(test1):
                        Nbr = Nbr + 1

    # calculating Nhheqdelta
    Nhheqdelta = Ncl + Nbr

    # calculating the parameter Ncyanideeqdelta
    patt_cyanide = Chem.MolFromSmarts('C#N')
    patt_tuple = mol.GetSubstructMatches(patt_cyanide)

    Ncyanideeqdelta = 0
    for j in range(0, len(patt_tuple)):
        atom = mol.GetAtomWithIdx(patt_tuple[j][0])
        z = atom.GetAtomicNum()
        if z != 6:
            atom = mol.GetAtomWithIdx(patt_tuple[j][1])
            z = atom.GetAtomicNum()
        if (z == 6):  # atomic number of Carbon belonging to cyanide group
            ii = atom.GetIdx(
            )  # atom index of Carbon belonging to cyanide group
            for i in range(
                    0, len(bondidxs)
            ):  # find the atom bonded to Carbon of cyanide group
                if (bondidxs[i][0] == ii):
                    atom_m = mol.GetAtomWithIdx(bondidxs[i][1])
                    z_m = atom_m.GetAtomicNum()
                    if z_m != 7:
                        jj = bondidxs[i][
                            1]  # index of atom bonded to Carbon of cyanide group
                if (bondidxs[i][1] == ii):
                    atom_m = mol.GetAtomWithIdx(bondidxs[i][0])
                    z_m = atom_m.GetAtomicNum()
                    if z_m != 7:
                        jj = bondidxs[i][
                            0]  # index of atom bonded to Carbon of cyanide group
            atom_n = mol.GetAtomWithIdx(jj)
            z_n = atom_n.GetAtomicNum()
            if (z_n == 6):
                if (deltaf[jj] == deltavf[jj]):
                    tmpvector1 = []
                    for i in range(
                            0, len(bondidxs)
                    ):  # find the atoms bonded to the carbon atom which is bonded to cyanide group
                        if (bondidxs[i][0] == jj or bondidxs[i][1] == jj):
                            tmpvector1.append(bondidxs[i])
                    tmpvector2 = []
                    for i in range(0, len(tmpvector1)):
                        if (tmpvector1[i][0] != beginatom
                                and tmpvector1[i][1] != endatom):
                            bondtmp = mol.GetBondBetweenAtoms(
                                tmpvector1[i][0], tmpvector1[i][1])
                            nbondtype = bondtmp.GetBondTypeAsDouble()
                            if nbondtype == 1:  # it is a single bond
                                tmpvector2.append(1)
                    test1 = np.isin(tmpvector2, 1)
                    if np.all(test1):
                        Ncyanideeqdelta = Ncyanideeqdelta + 1

    # calculating Nhbar
    Naromatic = rdcmd.CalcNumAromaticRings(mol)
    # Never Used
    # patt_aromatic = Chem.MolFromSmarts('a')
    # patt_tuple_aromatic = molp.GetSubstructMatches(patt_aromatic)
    patt_hydroxyl = Chem.MolFromSmarts('[OX2H]')
    patt_tuple_hydroxyl = molp.GetSubstructMatches(patt_hydroxyl)
    patt_alcohol = Chem.MolFromSmarts('[#6][OX2H]')
    patt_tuple_alcohol = molp.GetSubstructMatches(patt_alcohol)
    patt_carboxylic_acid = Chem.MolFromSmarts('[OX2H][CX3]=[OX1]')
    patt_tuple_carboxylic_acid = molp.GetSubstructMatches(patt_carboxylic_acid)
    # patt_sulfonic_acid = Chem.MolFromSmarts('[$([#16X4](=[OX1])(=[OX1])([#6])[OX2H,OX1H0-]),
    # $([#16X4+2]([OX1-])([OX1-])([#6])[OX2H,OX1H0-])]')
    patt_sulfonic_acid = Chem.MolFromSmarts('S(=O)(=O)O')
    patt_tuple_sulfonic_acid = molp.GetSubstructMatches(patt_sulfonic_acid)
    patt_amide = Chem.MolFromSmarts('[NX3][CX3](=[OX1])[#6] ')
    patt_tuple_amide = molp.GetSubstructMatches(patt_amide)
    patt_urea = Chem.MolFromSmarts('O=C(N)(N)')
    patt_tuple_urea = molp.GetSubstructMatches(patt_urea)

    Nhbar = 0
    condition = 0
    if Naromatic != 0:
        if (len(patt_tuple_amide) != 0 and condition == 0):
            Nhbar = Nhbar + 2 * Naromatic
            condition = 1
        if (len(patt_tuple_urea) != 0 and condition == 0):
            Nhbar = Nhbar + 2 * Naromatic
            condition = 1
        if (len(patt_tuple_alcohol) != 0):
            tmpvector1 = []
            tmpvector1 = np.intersect1d(
                patt_tuple_alcohol, patt_tuple_carboxylic_acid
            )  # looking if alcohol group is part of carboxylic acid group
            if len(tmpvector1) == 0:
                Nhbar = Nhbar + 1
                # there are alcohol groups and aromatic rings in the monomeric unit
        if (len(patt_tuple_hydroxyl) != 0):
            tmpvector2 = []
            tmpvector2 = np.intersect1d(
                patt_tuple_hydroxyl, patt_tuple_carboxylic_acid
            )  # looking if hydroxyl groups are part of carboxylic acid groups
            tmpvector3 = []
            tmpvector3 = np.intersect1d(
                patt_tuple_hydroxyl, patt_tuple_sulfonic_acid
            )  # looking if hydroxyl groups are part of sulfonic acid groups
            tmpvector4 = [
            ]  # storage all the intersections between carboxylic and sulfonic acid groups
            for i in range(0, len(tmpvector2)):
                tmpvector4.append(tmpvector2[i])
            for i in range(0, len(tmpvector3)):
                tmpvector4.append(tmpvector3[i])
            tmpvector5 = []
            tmpvector5 = np.setdiff1d(patt_tuple_hydroxyl, tmpvector4)
            # storage the hydroxyl groups which doesn't belong to carboxylic acid and sulfonic acid
            if (len(tmpvector5) != 0 and condition == 0):
                Nhbar = Nhbar + 2 * Naromatic
                condition = 1

    # calculating Nperf
    Nperf = 2 * Ncdouble - 14 * Nbbester + 5 * x4_Nper - \
        7 * Nhheqdelta - 6 * Ncyanideeqdelta - 12 * Nhbar

    return Nperf


def RingsFunction(mol):
    #
    # This function return a tuple (ringstuple) with atom indexes belonging to
    # all the rings of a molecule mol. To obtain the number of rings in a molecule
    # just use len(ringstuple)
    #
    # Finding rings at molecule m
    ri = mol.GetRingInfo()
    ringstuple = ri.AtomRings()
    return ringstuple


def FusedRingsFunction(mol):
    #
    # This function return a tuple (fusedtupleindex) with the ring indexes belonging to
    # all the fused rings of a molecule mol. To obtain the number of fused rings in a molecule
    # just use len(fusedtupleindex)
    #
    # Finding fused rings
    ringstuple = RingsFunction(mol)
    nrings = len(ringstuple)
    fusedtupleindex = np.ones([100], dtype=np.int64)
    fusedtupleindex = -1 * fusedtupleindex
    fused = np.zeros([100, 2], dtype=np.int64)
    nn = 0
    for i in range(0, nrings):
        for k in range(0, nrings):
            if (k > i):
                for j in range(0, len(ringstuple[i])):
                    for n in range(0, len(ringstuple[k])):
                        if (ringstuple[i][j] == ringstuple[k][n]):
                            fusedtupleindex[nn] = i
                            fused[nn][0] = i
                            fused[nn][1] = i
                            nn = nn + 1
                            fusedtupleindex[nn] = k
                            fused[nn][0] = k
                            fused[nn][1] = i
                            nn = nn + 1

    fusedtupleindex = np.sort(fusedtupleindex)
    fusedtupleindex = np.unique(fusedtupleindex)
    fusedtupleindex = np.flip(fusedtupleindex)
    fusedtupleindex = fusedtupleindex[fusedtupleindex > -1]

    return fusedtupleindex


def GroupFusedRingsFunction(mol):
    #
    # This function return a tuple (fusedgrouptupleindex) with the ring indexes belonging to
    # all the fused rings of a molecule mol. To obtain the number of fused rings in a molecule
    # just use len(fusedgrouptupleindex)
    #
    # Finding fused rings
    ringstuple = RingsFunction(mol)
    nrings = len(ringstuple)
    fusedtupleindex = np.ones([100], dtype=np.int64)
    fusedtupleindex = -1 * fusedtupleindex
    fused = np.zeros([100, 2], dtype=np.int64)
    nn = 0
    for i in range(0, nrings):
        for k in range(0, nrings):
            if (k > i):
                for j in range(0, len(ringstuple[i])):
                    for n in range(0, len(ringstuple[k])):
                        if (ringstuple[i][j] == ringstuple[k][n]):
                            fusedtupleindex[nn] = i
                            fused[nn][0] = i
                            fused[nn][1] = i
                            nn = nn + 1
                            fusedtupleindex[nn] = k
                            fused[nn][0] = k
                            fused[nn][1] = i
                            nn = nn + 1

    # Finding groups of fused rings
    fusedgroup = np.ones([100, 100], dtype=np.int64)
    fusedgroup = -1 * fusedgroup
    fusedgroupvector = np.ones([100], dtype=np.int64)
    fusedgroupvector = -1 * fusedgroupvector
    fusedgroupvectorunique = np.zeros([100], dtype=np.int64)
    fusedgrouptupleindex = []
    ii = 0
    jj = 0
    for i in range(0, nrings):
        kk = 0
        for j in range(0, nn):
            if (fused[j][1] == i):
                kk = 1
                if (fused[j][1] != fused[j][0]):
                    fusedgroup[ii][jj] = fused[j][0]
                    jj = jj + 1
                    fusedgroup[ii][jj] = fused[j][1]
                    jj = jj + 1
        if (kk == 0 and jj != 0):
            ii = ii + 1
            jj = 0

    for i in range(0, ii):
        fusedgroupvector = np.ones([100], dtype=np.int64)
        fusedgroupvector = -1 * fusedgroupvector
        for j in range(0, nn):
            if (fusedgroup[i][j] != -1):
                fusedgroupvector[j] = fusedgroup[i][j]
        fusedgroupvectorunique = np.unique(fusedgroupvector)
        fusedgroupvectorunique = fusedgroupvectorunique[
            fusedgroupvectorunique > -1]
        fusedgrouptupleindex.append(fusedgroupvectorunique)

    mergedlist = []
    mergedlistnew = []
    for i in range(0, len(fusedgrouptupleindex)):
        for j in range(0, len(fusedgrouptupleindex)):
            if j > i:
                test = np.isin(fusedgrouptupleindex[i],
                               fusedgrouptupleindex[j])
                if np.any(test):
                    mergedlist = np.concatenate(
                        (fusedgrouptupleindex[i], fusedgrouptupleindex[j]))
                    mergedlist = np.unique(mergedlist)
                    mergedlistnew.append(mergedlist)
                else:
                    test = np.isin(mergedlistnew, fusedgrouptupleindex[i])
                    if not np.any(test):
                        mergedlistnew.append(fusedgrouptupleindex[i])
                    if i == len(fusedgrouptupleindex) - 2:
                        mergedlistnew.append(fusedgrouptupleindex[i + 1])

    if len(fusedgrouptupleindex) == 1:
        mergedlistnew.append(fusedgrouptupleindex[0])

    return mergedlistnew


def RigidRingFunction(mol):
    #   This function return a tuple with the ring indexes belonging to
    #   distinct rigid rings (i.e., a rigid ring not fused to any other ring)
    # ---------------------------------------------------------------------
    ringstuple = RingsFunction(mol)
    nrings = len(ringstuple)
    fusedtupleindex = FusedRingsFunction(mol)
    nfusedrings = len(fusedtupleindex)

    rigidring = np.ones([100], dtype=np.int64)
    rigidring = -1 * rigidring
    qq = 0
    for i in range(0, nrings):
        pp = 0
        for j in range(0, nfusedrings):
            if (i == fusedtupleindex[j]):
                pp = 1
        if (pp == 0):
            rigidring[qq] = i
            qq = qq + 1
    rigidring = np.unique(rigidring)
    rigidring = rigidring[rigidring > -1]
    return rigidring


def MoleculeFunction(mol):
    #  This function gives all the atom indexes of molecule mol

    moleculetuple = []
    for atom in mol.GetAtoms():
        kk = atom.GetIdx()
        moleculetuple.append(kk)

    return moleculetuple


def SideGroupFunction(mol, lin):
    #  This function gives all the atom indexes belonging to the side group

    moleculetuple = MoleculeFunction(mol)
    backbonetuple = BackboneFunction(mol, lin)
    sidegrouptuple = []
    sidegrouptuple = np.setdiff1d(moleculetuple, backbonetuple)
    return sidegrouptuple


def molvtempFunction(mol, molar_volume, Tg, temperature):
    #  This function calculates the molar volume for a temperature T
    #  T (in Kelvin) different from roon temperature (298 K)

    molar_volume_tempf = 0
    alfa_r = 1 / (298 + 4.23 * Tg)

    if (temperature <= Tg and Tg >= 298):
        molar_volume_tempf = molar_volume * (
            1.42 * Tg + 0.15 * temperature) / (1.42 * Tg + 44.7)

    if (temperature > Tg and Tg >= 298):
        molar_volume_tempf = molar_volume * (1.57 * Tg + 0.30 *
                                             (temperature - Tg)) / (1.42 * Tg +
                                                                    44.7)

    if (temperature < Tg and Tg < 298):
        molar_volume_tempf = ((0.15 * molar_volume) * (temperature - Tg)) / (
            1.42 * Tg + 44.7) + molar_volume * (1 + alfa_r * (Tg - 298))

    if (temperature >= Tg and Tg < 298):
        molar_volume_tempf = molar_volume * (1 + alfa_r * (temperature - 298))

    return molar_volume_tempf


def FloryHugginsFunction(mol, Solb_w, Solb_p, temperature):
    #   This function calculates the Flory-Huggins parameter (fh_parameter)
    #   according to equations:
    #   2.41 and 2.42 and the conditions from Fig 2.18 of the text book:
    #
    # Polymer Solutions: An Introduction to Physical Properties.
    # Iwao Teraoka
    # Copyright © 2002 John Wiley & Sons, Inc.
    # ISBNs: 0-471-38929-3 (Hardback); 0-471-22451-0 (Electronic)
    #
    # fh_parameter can be interpreted as follow:
    #
    # If fh_parameter < 0.5 then water is a good solvent fo the polymer
    # If fh_parameter = 0.5 then water is in theta condition
    # If fh_parameter > 0.5 the water is NOT a good solvent for the polymer
    #
    # --------------------------------------------------------------------
    fh_parameterf = 0.0
    vmolar_water = 18.0  # Molar volume of water in cm^3/mol
    Navogrado = 6.02214076e23  # Avogrado number in mol^−1
    kb = 1.380649e-23  # Boltzman Constant in J.K^-1
    # Solubility parameters Solb_w and Solb_p are in (J/cm^3)^0.5

    fh_parameterf = (vmolar_water *
                     (Solb_w - Solb_p)**2) / (Navogrado * kb *
                                              temperature) + 0.34

    return fh_parameterf


def solutionviscosityFunction(mol, polymer_concentration_wt, fh_parameter, Mn,
                              molar_mass, water_viscosity, lin, Solb_w, Solb_p,
                              density_monomer_T):
    # ---------------------------------------------------------------------
    #   This function calculates the zero shear polymer solution
    #   viscosity (low concentration) in water for a specific polymer
    #   concentration (in weigth percentage).
    #
    #   water is the solvent
    #   polymer is the solute
    #   water + polymer are the solution
    #
    #   this is function is valid if the concetration is less or equal to
    #   1 wt% or 0.01 g/cm^3
    # ---------------------------------------------------------------------
    v_viscosityf = np.zeros(2, dtype=np.float64)
    solutionviscosityf = 0  # viscosity of water + polymer
    intr_viscosity = 0
    beginatom = int(lin[2])
    endatom = int(lin[3])
    shortestpathtuple = Chem.rdmolops.GetShortestPath(mol, beginatom, endatom)
    v_connec_indexf = ConnectivityIndex(mol, lin)
    x0v = v_connec_indexf[1]

    # calculating the number of hydrogens
    number_hydrogen = 0
    for atom in mol.GetAtoms():
        number_hydrogen = atom.GetImplicitValence() + number_hydrogen
        z = atom.GetAtomicNum()
        if z == 14:
            number_hydrogen = number_hydrogen + 1  # problem of rdkit for [Si]
    # Subtracted two hydrogens due to head and tail periodic monomeric bond
    number_hydrogen = number_hydrogen - 2

    # calculating the number of atoms in the shortest path
    Nsp = len(shortestpathtuple)

    # calculating J according to Eq(12.29) from Jocef Bicerano's book: J = - 1.9546*NH + 7.7686*x0v
    # it is the simplest implementation of J. There is other implementation for J according to
    # Eq(12.31)
    J = -1.9546 * number_hydrogen + 7.7686 * x0v

    # calculating K according to Eq()12.11) from Jocef Bicerano's book:
    K = J + 4.2 * Nsp

    # converting polymer concentration in wt% in g/cm^3:
    #  assuming that the water where polymer is dissolved always has 1g/cm^3 density
    # (environmental conditions):
    polymer_concentration = polymer_concentration_wt / 100

    # calculating the Mark-Houwink exponent "a" according to equations 12.12 and 12.13 from
    # Josef Bicerano's book
    if abs(Solb_w - Solb_p) <= 3:
        # a = 0.8 -0.1*(abs(Solb_w - Solb_p))  # original expression
        a = 0.8 - 0.025 * (abs(Solb_w - Solb_p))
    else:
        # a = 0.5   # original parameter
        a = 0.6

    if fh_parameter == 0.5:
        intr_viscosity = (Mn**0.5) * (K / molar_mass)**2
    else:
        Mcr = 169 * (
            molar_mass /
            K)**4  # critical polymer molecular weigth according to eq. 13.7
        intr_viscosity = 0.99328 * (
            ((K / molar_mass)**2) *
            (math.exp(8.5 * a**10.3)) * Mn**a) / (Mcr**(a - 0.5))

    solvate_vf = (0.4 * intr_viscosity * polymer_concentration) / (
        1 + 0.765 * intr_viscosity * polymer_concentration -
        1.91 * polymer_concentration / density_monomer_T)  # eq. 12.15
    solutionviscosityf = water_viscosity / (
        1 - 2.5 * solvate_vf + 11 * solvate_vf**5 - 11.5 * solvate_vf**7
    )  # eq. 12.16

    v_viscosityf[0] = intr_viscosity
    v_viscosityf[1] = solutionviscosityf
    return v_viscosityf


def temperature_half_decomposition_Function(mol, lin, molar_mass):
    # This function calculates the temperature of half decomposition
    # of the polymer according to equation 16.5 of the book:
    #
    # Prediction of Polymer Properties, third edition
    # Josef Bicerano
    #
    # The temperature of half decomposition is defined as
    # "the temperature at which the loss of weight during pyrolysis (at a constant
    #  rate of temperature rise) reaches 50% of its final value"

    Td_half = 0
    v_connec_indexf = ConnectivityIndex(mol, lin)
    x1v = v_connec_indexf[3]
    natom = mol.GetNumAtoms()

    # calculating the number of hydrogens
    number_hydrogen = 0
    for atom in mol.GetAtoms():
        number_hydrogen = atom.GetImplicitValence() + number_hydrogen
        z = atom.GetAtomicNum()
        if z == 14:
            number_hydrogen = number_hydrogen + 1  # problem of rdkit for [Si]
    # Subtracted two hydrogens due to head and tail periodic monomeric bond
    number_hydrogen = number_hydrogen - 2

    Yd_half = 7.17 * natom - 2.31 * number_hydrogen + 12.52 * x1v
    Td_half = 1000 * (Yd_half / molar_mass)

    return Td_half


def mechanical_properties_Function(mol, lin, temperature, Tg, Ecoh1,
                                   molar_volume):
    # This function calculates the mechanical properties of the
    # polymer: bulk modulus B, Young's modulus E, Shear modulus G,
    # and the Poisson's ratio nu

    v_mechpropf = np.zeros(5, dtype=np.float64)
    beginatom = int(lin[2])
    endatom = int(lin[3])
    v_connec_indexf = ConnectivityIndexOrig(mol, lin)
    x0 = v_connec_indexf[0]
    x1v = v_connec_indexf[3]
    V_0 = molvtempFunction(mol, molar_volume, Tg, 0)
    V_T = molvtempFunction(mol, molar_volume, Tg, temperature)

    # van der Waals Volume according to eq(3.9) from Josef Bicerano's book
    Vw = 3.861803 * x0 + 13.748435 * x1v

    # calculating the distance between head and tail atoms
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    dist = Chem.rdMolTransforms.GetBondLength(
        conf, beginatom,
        endatom) + 1.25  # sum 1.25 A due to the bond between monomeric units
    lm = dist * 1e-8  # converting Angstroms to centimeters

    # calculating the Poisson's ratio [dimensionless]
    nu = 0.513 - 3.054e-6 * math.sqrt(
        Vw / lm
    )  # Poisson's ration according to eq(11.10) from Josef Bicerano's book
    nu_0 = nu - (14900 / Tg) * (0.00163 + math.exp(
        0.459 *
        (285 - Tg)))  # eq.(11.11) from Josef Bicerano's book third edition
    nu_T = nu_0 + (50 * temperature / Tg) * (
        0.00163 + math.exp(0.459 * (temperature - Tg - 13))
    )  # eq.(11.12) from Josef Bicerano's book third edition

    # calculating the Bulk modulus [Mpa]
    B_T = 8.23333 * Ecoh1 * (5 * ((V_0**4) / (V_T**5)) - 3 * (
        (V_0**2) /
        (V_T**3)))  # eq.(11.13) from Josef Bicerano's book third edition

    # calculating the Young's modulus [Mpa]
    E_T = 3 * (1 - 2 * nu_T
               ) * B_T  # eq.(11.7) from Josef Bicerano's book third edition

    # calculating the Shear modulus [Mpa]
    G_T = E_T / (2 * (1 + nu_T)
                 )  # eq.(11.7) from Josef Bicerano's book third edition

    # calculating the brittle fracture stress sigma_f
    sigma_f = 2.288424 * 1e11 * lm / V_T

    # calculating the yield stress in uniaxial tension sigma_y
    sigma_y = 0.028 * E_T

    v_mechpropf[0] = B_T
    v_mechpropf[1] = E_T
    v_mechpropf[2] = G_T
    v_mechpropf[3] = sigma_f
    v_mechpropf[4] = sigma_y

    return v_mechpropf


def physical_crosslink_Function(mol):
    # This function gives some indexes which are related with the
    # physical polymer crosslink through hydrogen bonding and
    # ionic bond
    # The indexes are:
    # 1) Number of hydrogen bonding sites
    # 2) Charge of counterion which will bond to the ion in the
    #    polymer chain

    v_crosslink_indexf = np.zeros(2, dtype=np.int32)
    n_hbf = 0
    counterion_chargef = 0

    # Number of hydrogen bonding sites
    patt_hbonding = Chem.MolFromSmarts(
        '[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0]),$([O,S;H1;v2]-[!$(*=[O,N,P,S])'
        + ']),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]'
    )
    patt_tuple = mol.GetSubstructMatches(patt_hbonding)
    n_hbf = len(patt_tuple)

    # Charge of counterion
    for atom in mol.GetAtoms():
        n = atom.GetFormalCharge()
        if n == -1:
            counterion_chargef = 2
        if n == 1:
            counterion_chargef = -2
        if n == -2:
            counterion_chargef = 4
        if n == 2:
            counterion_chargef = -4

    v_crosslink_indexf[0] = n_hbf
    v_crosslink_indexf[1] = counterion_chargef

    return v_crosslink_indexf


def PolymerChainLengthFunction(mol, lin, Mn):
    # This function calculates the polymer chain length in nanometer

    beginatom = int(lin[2])
    endatom = int(lin[3])
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    dist = Chem.rdMolTransforms.GetBondLength(
        conf, beginatom,
        endatom) + 1.25  # sum 1.25 A due to the bond between monomeric units
    length = dist * 1e-1  # converting Angstroms to nanometer
    molar_mass = rdcd.ExactMolWt(
        mol
    ) - 2.016  # Subtracted two hydrogens due to head and tail periodic monomeric bond
    n_monomer = Mn / molar_mass
    chain_lengthf = int(length * n_monomer)
    return chain_lengthf


def ConvertOpsinToMolSmiles(smiles_string):
    ###########################################################################
    #
    # This function converts the OPSIN: Open Parser for Systematic IUPAC
    # nomenclature SMILES code into molecule SMILES code
    #
    # OPSIN: https://opsin.ch.cam.ac.uk/
    #
    ##########################################################################
    cont = 0
    for i in range(0, len(smiles_string)):
        if (smiles_string[i] == '[' and smiles_string[i + 1] == '*'):
            if cont == 0:
                begin1 = i
                end1 = i + 4
                cont = cont + 1
    for i in range(begin1 + 1, len(smiles_string)):
        if (smiles_string[i] == '[' and smiles_string[i + 1] == '*'):
            if cont == 1:
                begin2 = i
                end2 = i + 4
                cont = cont + 1

    smiles_molf = smiles_string[:(begin1)] + smiles_string[
        (end1 + 1):(begin2)] + smiles_string[(end2 + 1):]
    return smiles_molf


def BackboneFunction(mol, lin):
    #  This function determines the atom indexes belonging to the backbone
    #  polymer chain

    beginatom = int(lin[2])
    endatom = int(lin[3])
    shortestpathtuple = Chem.rdmolops.GetShortestPath(mol, beginatom, endatom)

    ringstuple = RingsFunction(mol)

    backbonetuple = []
    for i in range(0, len(shortestpathtuple)):
        backbonetuple.append(shortestpathtuple[i])

    if len(ringstuple) > 0:
        for i in range(0, len(ringstuple)):
            test1 = np.isin(ringstuple[i], shortestpathtuple)
            ii = 0
            for k in range(0, len(test1)):
                if test1[k]:
                    ii = ii + 1
            if ii > 1:
                for j in range(0, len(ringstuple[i])):
                    backbonetuple.append(ringstuple[i][j])

    backbonetuple = np.unique(backbonetuple)
    return backbonetuple


def BackboneRingFunction(mol, lin):
    #  This function determines the ring indexes belonging to the backbone
    #  polymer chain

    beginatom = int(lin[2])
    endatom = int(lin[3])
    shortestpathtuple = Chem.rdmolops.GetShortestPath(mol, beginatom, endatom)

    ringstuple = RingsFunction(mol)

    backboneringtuple = []

    if len(ringstuple) > 0:
        for i in range(0, len(ringstuple)):
            test1 = np.isin(ringstuple[i], shortestpathtuple)
            ii = 0
            for k in range(0, len(test1)):
                if test1[k]:
                    ii = ii + 1
            if ii > 1:
                backboneringtuple.append(i)

    return backboneringtuple


def BetaFunction(mol, lin):
    #
    #  calculating beta -> beta_ij = delta_i*delta_j
    #

    nbond = mol.GetNumBonds()
    betaf = np.zeros(nbond + 1, dtype=np.float64)
    bondf = BondFunction(mol, lin)
    deltaf = DeltaFunction(mol, lin)
    for i in range(0, nbond + 1):
        ii = bondf[i][0]
        jj = bondf[i][1]
        betaf[i] = deltaf[ii] * deltaf[jj]

    return betaf


def BetavFunction(mol, lin):
    #
    #  calculating betav -> betav_ij = deltav_i*deltav_j
    #

    nbond = mol.GetNumBonds()
    betavf = np.zeros(nbond + 1, dtype=np.float64)
    bondf = BondFunction(mol, lin)
    deltavf = DeltavFunction(mol, lin)
    for i in range(0, nbond + 1):
        ii = bondf[i][0]
        jj = bondf[i][1]
        betavf[i] = deltavf[ii] * deltavf[jj]

    return betavf


def BondFunction(mol, lin):
    #
    #   This function calculates the bonds between the atoms for
    #   molecule mol, obtained from SMILES
    #

    nbond = mol.GetNumBonds()
    bondf = np.zeros([nbond + 1, 2], dtype=np.int64)
    for i in range(0, nbond):
        bondf[i][0] = mol.GetBondWithIdx(i).GetBeginAtomIdx()
        bondf[i][1] = mol.GetBondWithIdx(i).GetEndAtomIdx()
    # defining the periodic bond between monomers
    bondf[nbond][0] = int(lin[2])
    bondf[nbond][1] = int(lin[3])

    return bondf


def ConnectivityIndex(mol, lin):
    #
    #   This function calculates the zero and first order connectivity
    #   indices of a polymer monomer. This is the first step to
    #   calculate polymer properties.
    #
    #   INPUT: m is a rdkit.Chem molecule class with informations based
    #          on SMILES code.
    #
    #   OUTPUT: x0, x0v, x1, x1v
    #
    #   Ronaldo Giro  6, February 2019

    v_connec_indexf = np.zeros(4, dtype=np.float64)
    natom = mol.GetNumAtoms()
    nbond = mol.GetNumBonds()
    # calculating x0 and x0v -> zero order connectivity indices
    x0 = 0
    x0v = 0
    deltaf = DeltaFunction(mol, lin)
    deltavf = DeltavFunction(mol, lin)
    for i in range(0, natom):
        x0 = x0 + 1 / math.sqrt(deltaf[i])
        x0v = x0v + 1 / math.sqrt(deltavf[i])

    # calculating x1 and x1v -> first order connectivity indices
    x1 = 0
    x1v = 0
    betaf = BetaFunction(mol, lin)
    betavf = BetavFunction(mol, lin)
    for i in range(0, nbond + 1):
        x1 = x1 + 1 / math.sqrt(betaf[i])
        x1v = x1v + 1 / math.sqrt(betavf[i])

    v_connec_indexf[0] = x0
    v_connec_indexf[1] = x0v
    v_connec_indexf[2] = x1
    v_connec_indexf[3] = x1v
    return v_connec_indexf


def ConnectivityIndexOrig(mol, lin):
    #
    #   This function calculates the zero and first order connectivity
    #   indices of a polymer monomer. This is the first step to
    #   calculate polymer properties.
    #
    #   INPUT: m is a rdkit.Chem molecule class with informations based
    #          on SMILES code.
    #
    #   OUTPUT: x0, x0v, x1, x1v
    #
    #   Ronaldo Giro  6, February 2019

    v_connec_indexf = np.zeros(4, dtype=np.float64)
    natom = mol.GetNumAtoms()
    nbond = mol.GetNumBonds()
    # calculating x0 and x0v -> zero order connectivity indices
    x0 = 0
    x0v = 0
    deltaf = DeltaFunction(mol, lin)
    deltavf = DeltavFunctionOrig(mol, lin)
    for i in range(0, natom):
        x0 = x0 + 1 / math.sqrt(deltaf[i])
        x0v = x0v + 1 / math.sqrt(deltavf[i])

    # calculating x1 and x1v -> first order connectivity indices
    x1 = 0
    x1v = 0
    betaf = BetaFunction(mol, lin)
    betavf = BetavFunction(mol, lin)
    for i in range(0, nbond + 1):
        x1 = x1 + 1 / math.sqrt(betaf[i])
        x1v = x1v + 1 / math.sqrt(betavf[i])

    v_connec_indexf[0] = x0
    v_connec_indexf[1] = x0v
    v_connec_indexf[2] = x1
    v_connec_indexf[3] = x1v
    return v_connec_indexf


def DeltaFunction(mol, lin):
    #
    #   This function calculates the number of non-hydrogen atoms to
    #   which a given non-hydrogen atom is bonded
    #

    natom = mol.GetNumAtoms()
    nbond = mol.GetNumBonds()
    deltaf = np.zeros(natom, dtype=np.float64)
    bondf = BondFunction(mol, lin)
    # calculating delta -> number of non-hydrogen atoms bonded to the non-hydrogen atom
    for i in range(0, natom):
        nn = 0
        for j in range(0, nbond + 1):
            if bondf[j][0] == i:
                nn = nn + 1
            if bondf[j][1] == i:
                nn = nn + 1
        deltaf[i] = nn

    return deltaf


def DeltavFunction(mol, lin):
    #
    #   This function calculates deltav, the atom connectivity index
    #   associated with the electronic environment
    #

    natom = mol.GetNumAtoms()
    nbond = mol.GetNumBonds()
    deltavf = np.zeros(natom, dtype=np.float64)
    nh = np.zeros(natom, dtype=np.int64)
    bondf = BondFunction(mol, lin)
    # calculating the number of hydrogens
    for atom in mol.GetAtoms():
        kk = atom.GetIdx()
        nh[kk] = atom.GetImplicitValence()
        z = atom.GetAtomicNum()
        if z == 14:
            nh[kk] = nh[kk] + 1
        if (kk == bondf[nbond][0] or kk == bondf[nbond][1]):
            nh[kk] = nh[kk] - 1

    # calculating deltav
    valence_electron = {
        6: 4,
        7: 5,
        8: 6,
        9: 7,
        14: 4,
        16: 6,
        17: 7,
        35: 7
    }  # dictionary relating atomic number : valence electron
    deltaf = DeltaFunction(mol, lin)
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        z = atom.GetAtomicNum()
        if z == 14:
            z = 6  # replacing Si -> C according to Table 2.3 Josef Bicerano's book
        zv = valence_electron[z]
        deltavf[i] = (zv - nh[i]) / (z - zv - 1)  # deltav definition
        atom_charge = atom.GetFormalCharge()
        if (z == 7 and atom_charge == 1):  # Nitrogen with +1 charge
            deltavf[i] = deltavf[i] + 1
        if (z == 8 and atom_charge == -1):  # Oxygen with -1 charge
            deltavf[i] = deltavf[i] - 1
        if (z == 16 and deltaf[i] == 3):  # oxidation state +4 of Sulfur
            deltavf[i] = deltavf[i] + 1
        if (z == 16 and deltaf[i] == 4):  # oxidation state +6 of Sulfur
            deltavf[i] = deltavf[i] + 2

    return deltavf


def DeltavFunctionOrig(mol, lin):
    #
    #   This function calculates deltav, the atom connectivity index
    #   associated with the electronic environment withou any modification
    #   for Si atom (I mean, replacing Si -> C as)
    #

    natom = mol.GetNumAtoms()
    nbond = mol.GetNumBonds()
    deltavf = np.zeros(natom, dtype=np.float64)
    nh = np.zeros(natom, dtype=np.int64)
    bondf = BondFunction(mol, lin)
    # calculating the number of hydrogens
    for atom in mol.GetAtoms():
        kk = atom.GetIdx()
        nh[kk] = atom.GetImplicitValence()
        z = atom.GetAtomicNum()
        if z == 14:
            nh[kk] = nh[kk] + 1
        if (kk == bondf[nbond][0] or kk == bondf[nbond][1]):
            nh[kk] = nh[kk] - 1

    # calculating deltav
    valence_electron = {
        6: 4,
        7: 5,
        8: 6,
        9: 7,
        14: 4,
        16: 6,
        17: 7,
        35: 7
    }  # dictionary relating atomic number : valence electron
    deltaf = DeltaFunction(mol, lin)
    for atom in mol.GetAtoms():
        i = atom.GetIdx()
        z = atom.GetAtomicNum()
        zv = valence_electron[z]
        deltavf[i] = (zv - nh[i]) / (z - zv - 1)  # deltav definition
        atom_charge = atom.GetFormalCharge()
        if (z == 7 and atom_charge == 1):  # Nitrogen with +1 charge
            deltavf[i] = deltavf[i] + 1
        if (z == 8 and atom_charge == -1):  # Oxygen with -1 charge
            deltavf[i] = deltavf[i] - 1
        if (z == 16 and deltaf[i] == 3):  # oxidation state +4 of Sulfur
            deltavf[i] = deltavf[i] + 1
        if (z == 16 and deltaf[i] == 4):  # oxidation state +6 of Sulfur
            deltavf[i] = deltavf[i] + 2

    return deltavf


def HeadTailAtoms(smiles_string):
    ###########################################################################
    #
    # This function gives the head and tail atom indexes from the OPSIN
    # SMILES strings
    #
    # OPSIN: Open Parser for Systematic IUPAC Nomenclature SMILES
    # https://opsin.ch.cam.ac.uk/
    #
    ###########################################################################
    atom_indexesf = np.zeros(2, dtype=np.int32)
    head_index = -1
    tail_index = -1
    chars = ('()[]=-+*:#0123456789ilr')

    # looking for the head marker for polymer monomeric unit: [*:1]
    for i in range(0, len(smiles_string)):
        if smiles_string[i] == '[':
            if smiles_string[i + 3] == '1':
                ii = i  # smiles_string index for the head atom

    # looking for the tail marker for polymer monomeric unit: [*:2]
    for i in range(0, len(smiles_string)):
        if smiles_string[i] == '[':
            if smiles_string[i + 3] == '2':
                jj = i  # smiles_string index for the tail atom

    # defining the head atom index of the monomeric polymer unit:
    pp = 0
    qq = 0
    cont = 0
    if ii == 0:
        head_index = 0
    else:
        if smiles_string[ii - 1] != ')':
            head_index = ii - 1
        else:
            for i in range(ii, 0, -1):
                if smiles_string[i] == ')':
                    pp = pp + 1
                if smiles_string[i] == '(':
                    qq = qq + 1
                if (pp == qq and cont == 0):
                    if (pp != 0 and qq != 0):
                        head_index = i - 1
                        cont = cont + 1

    ii = head_index
    pp = 0
    qq = 0
    cont = 0
    if ii == 0:
        head_index = 0
    else:
        if smiles_string[ii] != ')':
            head_index = ii
        else:
            for i in range(ii, 0, -1):
                if smiles_string[i] == ')':
                    pp = pp + 1
                if smiles_string[i] == '(':
                    qq = qq + 1
                if (pp == qq and cont == 0):
                    if (pp != 0 and qq != 0):
                        head_index = i - 1
                        cont = cont + 1

    # subtracting the symbols
    head_atom_index = head_index
    if head_atom_index > 0:
        sttmp = smiles_string[:(head_index + 1)]
        for elem in sttmp:
            if elem in chars:
                head_atom_index = head_atom_index - 1

    # defining the tail atom index of the monomeric polymer unit:
    pp = 0
    qq = 0
    cont = 0
    if smiles_string[jj - 1] != ')':
        tail_index = jj - 1
    else:
        for i in range(jj, 0, -1):
            if smiles_string[i] == ')':
                pp = pp + 1
            if smiles_string[i] == '(':
                qq = qq + 1
            if (pp == qq and cont == 0):
                if (pp != 0 and qq != 0):
                    tail_index = i - 1
                    cont = cont + 1

    jj = tail_index
    pp = 0
    qq = 0
    cont = 0
    if smiles_string[jj] != ')':
        tail_index = jj
    else:
        for i in range(jj, 0, -1):
            if smiles_string[i] == ')':
                pp = pp + 1
            if smiles_string[i] == '(':
                qq = qq + 1
            if (pp == qq and cont == 0):
                if (pp != 0 and qq != 0):
                    tail_index = i - 1
                    cont = cont + 1

    # subtracting the symbols
    tail_atom_index = tail_index
    sttmp = smiles_string[:tail_index + 1]
    for elem in sttmp:
        if elem in chars:
            tail_atom_index = tail_atom_index - 1

    atom_indexesf[0] = head_atom_index
    atom_indexesf[1] = tail_atom_index

    return atom_indexesf


def HeadTailAtomsMolext(mol, lin):
    # This function add the head and tail atoms of a molecule to a new
    # molecule in order to take into account the polymer periodicity
    # effect. Then, this function gives the head and tail atom indexes
    # from the OPSIN SMILES strings
    #
    # OPSIN: Open Parser for Systematic IUPAC Nomenclature SMILES
    # https://opsin.ch.cam.ac.uk/
    #
    ###########################################################################
    atom_indexesf2 = np.zeros(2, dtype=np.int32)

    # defining the molecule with periodic bonds to calculate fragments
    bi = int(lin[2])
    bf = int(lin[3])
    atomi = mol.GetAtomWithIdx(bi)
    atomf = mol.GetAtomWithIdx(bf)
    sti = atomi.GetSymbol()
    stf = atomf.GetSymbol()
    if sti == 'Si':
        sti = '[Si]'
    if stf == 'Si':
        stf = '[Si]'

    # replacing the head and tail ghost atoms [*:1] and [*:2] with the sti and stf atoms
    sttmp1 = lin[1]
    if (sttmp1[1] == '*' and sttmp1[3] == '1'):
        sttmp2 = sttmp1.replace("[*:1]", "[*:1]" + stf)
    else:
        sttmp2 = sttmp1.replace("[*:1]", stf + "[*:1]")
    st = sttmp2.replace("[*:2]", sti + "[*:2]")

    # returning the head and tail atom indexes of extended molecule
    atom_indexesf2 = HeadTailAtoms(st)
    return atom_indexesf2


def molextFunction(mol, lin):
    # This function add the head and tail atoms of a molecule to a new
    # molecule in order to take into account the polymer periodicity
    # effect. So, molpf is mol with two added atoms: the head and the
    # tail atoms from mol
    #
    # smiles_string is the OPSIN: Open Parser for Systematic IUPAC
    # nomenclature SMILES with the periodic markers: [*:1] and [*:2]
    # for head and tail ghost atoms
    #
    # defining the molecule with periodic bonds to calculate fragments
    bi = int(lin[2])
    bf = int(lin[3])
    atomi = mol.GetAtomWithIdx(bi)
    atomf = mol.GetAtomWithIdx(bf)
    sti = atomi.GetSymbol()
    stf = atomf.GetSymbol()
    if sti == 'Si':
        sti = '[Si]'
    if stf == 'Si':
        stf = '[Si]'

    # replacing the head and tail ghost atoms [*:1] and [*:2] with the sti and stf atoms
    sttmp1 = lin[1]
    sttmp2 = sttmp1.replace("[*:1]", stf)
    st = sttmp2.replace("[*:2]", sti)
    molpf = Chem.MolFromSmiles(st)
    return molpf


def molextFunctionOld(mol, lin):
    # This function add the head and tail atoms of a molecule to a new
    # molecule in order to take into account the polymer periodicity
    # effect. So, molpf is mol with two added atoms: the head and the
    # tail atoms from mol

    # defining the molecule with periodic bonds to calculate fragments
    bi = int(lin[2])
    bf = int(lin[3])
    atomi = mol.GetAtomWithIdx(bi)
    atomf = mol.GetAtomWithIdx(bf)
    sti = atomi.GetSymbol()
    stf = atomf.GetSymbol()
    if sti == 'Si':
        sti = '[Si]'
    if stf == 'Si':
        stf = '[Si]'
    st = stf + lin[1] + sti
    molpf = Chem.MolFromSmiles(st)
    return molpf


def x1NtgFunction(mol, lin):
    #   x1_Ntg is one of thirteen structural parameters which establishes
    #   a correlation with glass transition temperature Tg
    #
    #   FOR EACH RULE IMPLEMENTED, THE CONTRIBUTION OF X1_NTG IS TAKEN
    #   INTO ACCOUNT JUST ONE TIME. FOR EXAMPLE: IF THERE ARE TWO DISTINCT
    #   RIGID RING ALONG THE SHORTEST PATH ACROSS THE CHAIN BACKBONE IN A
    #   PARA BONDING CONFIGURATION IT WILL CONTRIBUT JUST ONCE, I.E
    #   X1_NTG = 1, INSTEAD OF 2 DUE TO THE TWO RING.

    x1_Ntg = 0
    molp = molextFunction(mol, lin)
    # Never Used
    # rigidring = RigidRingFunction(mol)
    ringstuple = RingsFunction(mol)
    fusedgrouptupleindex = GroupFusedRingsFunction(mol)
    backbonetuple = BackboneFunction(mol, lin)
    rigidringp = RigidRingFunction(molp)
    ringstuplep = RingsFunction(molp)
    fusedgrouptupleindexp = GroupFusedRingsFunction(molp)

    # Finding the shortest path
    beginatom = int(lin[2])
    endatom = int(lin[3])
    shortestpathtuple = Chem.rdmolops.GetShortestPath(mol, beginatom, endatom)
    atom_indexesf2 = np.zeros(2, dtype=np.int32)
    atom_indexesf2 = HeadTailAtomsMolext(mol, lin)
    beginatomp = int(atom_indexesf2[0])
    endatomp = int(atom_indexesf2[1])
    shortestpathtuplep = Chem.rdmolops.GetShortestPath(molp, beginatomp,
                                                       endatomp)

    # Finding the bonds between the atoms of molecule
    bondidxsp = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                 for b in molp.GetBonds()]
    bondidxsp.append((beginatom, endatom + 2))

    # x1_Ntg:
    # A distinct rigid ring (i.e., a rigid ring not fused to any other ring) along the shortest path
    # across the chain backbone contributes +1 to x1 if it is in a para bonding configuration, and
    # does not contribute anything to x1 if it is in a meta or ortho bonding configuration.

    patt_para = Chem.MolFromSmarts(
        '*-!:aaaa-!:* ')  # para bonding configuration
    patt_tuple = molp.GetSubstructMatches(patt_para)

    tmpvector1 = np.zeros([100], dtype=np.int64)
    tmpvector2 = np.zeros([100], dtype=np.int64)
    condition_x1 = 0
    for i in range(0, len(patt_tuple)):
        for j in range(0, len(rigidringp)):
            k = rigidringp[j]
            tmpvector1 = np.intersect1d(patt_tuple[i], ringstuplep[k])
            tmpvector2 = np.intersect1d(tmpvector1, shortestpathtuplep)
            if (len(tmpvector2) == 4 and condition_x1 == 0):
                x1_Ntg = x1_Ntg + 1
                condition_x1 = 1

    # x1_Ntg: If there is a rigid fused ring unit of n >= 2 rings, containing exactly two of its
    # rigid rings along
    # the shortest path across the chain backbone, the contribution of this rigid fused ring unit
    # to x1 is determined as follows:
    # (a) It contributes +2 if the two bonds connecting it to the rest of the polymer chain are
    # close to being collinear.
    # (b) It contributes only +1 if the two bonds connecting it to the rest of the chain are far
    # from being collinear, causing large chain displacements to occur for even a small rotation of
    # the unit around either
    # If a rigid fused ring unit contains n>=3 rings along the shortest path across the chain
    # backbone,
    # it contributes (n+1) to xj, regardless of the relative orientation of the two bonds
    # connecting it to the rest of the backbone.

    tmpvector = []
    condition_x1 = 0
    for i in range(0, len(fusedgrouptupleindexp)):
        if (len(fusedgrouptupleindexp[i]) == 2):  # two fused rings
            pp = fusedgrouptupleindexp[i][0]
            tmpvector = ringstuplep[pp]
            test1 = np.isin(tmpvector, shortestpathtuplep)
            pp = fusedgrouptupleindexp[i][1]
            tmpvector = ringstuplep[pp]
            test2 = np.isin(tmpvector, shortestpathtuplep)
            if (np.any(test1) and np.any(test2)):
                # these two fused rings are in the shortest path
                sideatomidxtmp1 = []
                sideatomidxtmp2 = []
                sideatomidxtmp3 = []
                sideatomidx = []
                for j in range(0, len(fusedgrouptupleindexp[i])):
                    pp = fusedgrouptupleindexp[i][j]
                    tmpvector = ringstuplep[pp]
                    for k in range(
                            0, len(bondidxsp)
                    ):  # looking for side atoms attached to the ring
                        test1 = np.isin(tmpvector, bondidxsp[k][0])
                        test2 = np.isin(tmpvector, bondidxsp[k][1])
                        if (np.any(test1) or np.any(test2)):
                            if not np.any(test1):
                                sideatomidxtmp1.append(bondidxsp[k][0])
                            if not np.any(test2):
                                sideatomidxtmp1.append(bondidxsp[k][1])
                qq = fusedgrouptupleindexp[i][0]
                rr = fusedgrouptupleindexp[i][1]
                sideatomidxtmp2 = np.setdiff1d(sideatomidxtmp1,
                                               ringstuplep[qq])
                sideatomidxtmp3 = np.setdiff1d(sideatomidxtmp2,
                                               ringstuplep[rr])
                sideatomidx = np.intersect1d(
                    shortestpathtuplep,
                    sideatomidxtmp3)  # shortest path atom bonded at fused ring
                sideatomidxtmp4 = []
                for k in range(0, len(bondidxsp)):
                    # looking for atom index bonded to the sideatomidx (side atom of fused ring
                    # belonging to the shortest path)
                    test1 = np.isin(sideatomidx[0], bondidxsp[k][0])
                    test2 = np.isin(sideatomidx[0], bondidxsp[k][1])
                    if (np.any(test1) or np.any(test2)):
                        if not np.any(test1):
                            sideatomidxtmp4.append(bondidxsp[k][0])
                        if not np.any(test2):
                            sideatomidxtmp4.append(bondidxsp[k][1])
                kk = np.intersect1d(sideatomidxtmp4, ringstuplep[qq])
                # it is the atom index belonging to the fused ring bonded to the side atom
                # andbelonging to the shortest path
                if len(kk) == 0:
                    kk = np.intersect1d(sideatomidxtmp4, ringstuplep[rr])
                    # it is the atom index belonging to the fused ring bonded to the side atom and
                    # belonging to the shortest path
                molt = molp
                AllChem.EmbedMolecule(molt, randomSeed=42)
                conf = molt.GetConformer(0)
                angle_degrees = Chem.rdMolTransforms.GetAngleDeg(
                    conf, int(sideatomidx[0]), int(kk[0]), int(sideatomidx[1]))
                if (angle_degrees > 165 and angle_degrees < 195):
                    # two bonds connecting fused ring to the rest of the polymer chain are close to
                    # being collinear
                    if condition_x1 == 0:
                        x1_Ntg = x1_Ntg + 2
                        condition_x1 = 1
                else:
                    # two bonds connecting fused ring to the rest of the polymer chain are far from
                    # being collinear
                    if condition_x1 == 0:
                        x1_Ntg = x1_Ntg + 1
                        condition_x1 = 1
        if (len(fusedgrouptupleindex[i]) >= 3):
            condition = 0
            for j in range(0, len(fusedgrouptupleindex[i])):
                pp = fusedgrouptupleindex[i][j]
                tmpvector = ringstuple[pp]
                test1 = np.isin(tmpvector, shortestpathtuple)
                if not np.any(
                        test1):  # this fused ring isn't in the shortest path
                    condition = 1
            if condition == 0:
                if condition_x1 == 0:
                    x1_Ntg = x1_Ntg + (len(fusedgrouptupleindex[i]) + 1)
                    condition_x1 = 1

    # x1_Ntg ->
    # A hydrogen-bonding moiety containing non-hydrogen atoms on the chain backbone
    # contributes +1 to x1_Ntg

    # patt_hbonding = Chem.MolFromSmarts('[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),
    # $([n;H1;+0]),$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),
    # $([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]')
    patt_hbonding = Chem.MolFromSmarts(
        '[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([S;H1;+0]),$([n;H1;+0]),$([S;H1;v2]-[!$(*=[O,N,P,S])])'
        + ',$([S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]'
    )
    patt_tuple = mol.GetSubstructMatches(patt_hbonding)

    nh_acceptor = np.intersect1d(
        patt_tuple, backbonetuple
    )  # eliminating all nh_acceptor which does not belong to the backbone

    # nh_acceptortmp = nh_acceptor
    # for i in range(0,len(ringstuple)):   # eliminating all nh_acceptor who belongs to a ring
    # nh_acceptor = np.setdiff1d(nh_acceptortmp,ringstuple[i])
    # nh_acceptortmp = nh_acceptor

    # print('nH_acceptor tuple which is not in a ring = ',nh_acceptor)

    if len(nh_acceptor) >= 1:
        # x1_Ntg = x1_Ntg + len(nh_acceptor)
        x1_Ntg = x1_Ntg + 1

    return x1_Ntg


def x2NtgFunction(mol, lin):
    #   x2_Ntg is one of thirteen structural parameters which establishes
    #   a correlation with glass transition temperature Tg

    x2_Ntg = 0
    molp = molextFunction(mol, lin)
    rigidringp = RigidRingFunction(molp)
    ringstuplep = RingsFunction(molp)
    fusedgrouptupleindexp = GroupFusedRingsFunction(molp)

    # Finding the shortest path
    # never used
    # beginatom = int(lin[2])
    # endatom = int(lin[3])
    # shortestpathtuple = Chem.rdmolops.GetShortestPath(mol, beginatom, endatom)
    atom_indexesf2 = np.zeros(2, dtype=np.int32)
    atom_indexesf2 = HeadTailAtomsMolext(mol, lin)
    beginatomp = int(atom_indexesf2[0])
    endatomp = int(atom_indexesf2[1])
    shortestpathtuplep = Chem.rdmolops.GetShortestPath(molp, beginatomp,
                                                       endatomp)

    # x2_Ntg: Each distinct rigid ring, and each fused ring unit containing two rigid rings along
    # the shortest
    # path across the chain backbone, located in a meta or equivalent bonding configuration,
    # contributes +1 to x
    # The equivalent meta bonding configuration was not implemented

    patt_meta = Chem.MolFromSmarts(
        '*-!:aaa-!:* ')  # meta bonding configuration
    patt_tuple = molp.GetSubstructMatches(patt_meta)

    # each distinct rigid ring
    tmpvector1 = np.zeros([100], dtype=np.int64)
    tmpvector2 = np.zeros([100], dtype=np.int64)
    tmpvector3 = np.zeros([100], dtype=np.int64)
    for i in range(0, len(patt_tuple)):
        for j in range(0, len(rigidringp)):
            k = rigidringp[j]
            tmpvector1 = np.intersect1d(patt_tuple[i], ringstuplep[k])
            tmpvector2 = np.intersect1d(tmpvector1, shortestpathtuplep)
            tmpvector3 = np.intersect1d(patt_tuple[i], shortestpathtuplep)
            if (len(tmpvector2) == 3 and len(tmpvector3) == 5):
                x2_Ntg = x2_Ntg + 1
    # each fused ring unit containing two rigid rings along the shortest path across the chain
    # backbone
    tmpvector1 = np.zeros([100], dtype=np.int64)
    tmpvector2 = np.zeros([100], dtype=np.int64)
    tmpvector3 = np.zeros([100], dtype=np.int64)
    for i in range(0, len(patt_tuple)):
        for j in range(0, len(fusedgrouptupleindexp)):
            for k in range(0, len(fusedgrouptupleindexp[j])):
                p = fusedgrouptupleindexp[j][k]
                tmpvector1 = np.intersect1d(patt_tuple[i], ringstuplep[p])
                tmpvector2 = np.intersect1d(tmpvector1, shortestpathtuplep)
                tmpvector3 = np.intersect1d(patt_tuple[i], shortestpathtuplep)
                if (len(tmpvector2) == 3 and len(tmpvector3) == 5):
                    if len(fusedgrouptupleindexp[j]) == 2:
                        x2_Ntg = x2_Ntg + 1

    # x2_Ntg ->
    # Each ortho isolated rigid ring on the shortest path across the chain backbone contributes +5

    patt_ortho = Chem.MolFromSmarts('*-!:aa-!:* ')  # ortho
    patt_tuple = molp.GetSubstructMatches(patt_ortho)

    tmpvector1 = np.zeros([100], dtype=np.int64)
    tmpvector2 = np.zeros([100], dtype=np.int64)
    tmpvector3 = np.zeros([100], dtype=np.int64)
    for i in range(0, len(patt_tuple)):
        for j in range(0, len(rigidringp)):
            k = rigidringp[j]
            tmpvector1 = np.intersect1d(patt_tuple[i], ringstuplep[k])
            tmpvector2 = np.intersect1d(tmpvector1, shortestpathtuplep)
            tmpvector3 = np.intersect1d(patt_tuple[i], shortestpathtuplep)
            if (len(tmpvector2) == 2 and len(tmpvector3) == 4):
                x2_Ntg = x2_Ntg + 5

    # x2_Ntg ->
    # Each two-ring fused rigid ring unit along the shortest path across the chain backbone in an
    # ortho or equivalent bonding configuration contributes +8 to x2.
    # The equivalent ortho bonding configuraiton was not implemented

    patt_ortho = Chem.MolFromSmarts('*-!:aa-!:* ')  # ortho
    patt_tuple = molp.GetSubstructMatches(patt_ortho)

    tmpvector1 = np.zeros([100], dtype=np.int64)
    tmpvector2 = np.zeros([100], dtype=np.int64)
    tmpvector3 = np.zeros([100], dtype=np.int64)
    for i in range(0, len(patt_tuple)):
        for j in range(0, len(fusedgrouptupleindexp)):
            for k in range(0, len(fusedgrouptupleindexp[j])):
                p = fusedgrouptupleindexp[j][k]
                tmpvector1 = np.intersect1d(patt_tuple[i], ringstuplep[p])
                tmpvector2 = np.intersect1d(tmpvector1, shortestpathtuplep)
                tmpvector3 = np.intersect1d(patt_tuple[i], shortestpathtuplep)
                if (len(tmpvector2) == 2 and len(tmpvector3) == 4):
                    if len(fusedgrouptupleindexp[j]) == 2:
                        x2_Ntg = x2_Ntg + 8

    return x2_Ntg


def x3NtgFunction(mol, lin):
    #   x3_Ntg is one of thirteen structural parameters which establishes
    #   a correlation with glass transition temperature Tg

    x3_Ntg = 0
    backboneringtuple = BackboneRingFunction(mol, lin)
    ringstuple = RingsFunction(mol)
    fusedtupleindex = FusedRingsFunction(mol)
    fusedgrouptupleindex = GroupFusedRingsFunction(mol)
    beginatom = int(lin[2])
    endatom = int(lin[3])
    shortestpathtuple = Chem.rdmolops.GetShortestPath(mol, beginatom, endatom)
    backbonetuple = BackboneFunction(mol, lin)
    molp = molextFunction(mol, lin)
    ringstuplep = RingsFunction(molp)
    atom_indexesf2 = np.zeros(2, dtype=np.int32)
    atom_indexesf2 = HeadTailAtomsMolext(mol, lin)
    beginatomp = int(atom_indexesf2[0])
    endatomp = int(atom_indexesf2[1])
    shortestpathtuplep = Chem.rdmolops.GetShortestPath(molp, beginatomp,
                                                       endatomp)
    linf = []
    linf.append(' ')
    linf.append(' ')
    linf.append(str(beginatomp))
    linf.append(str(endatomp))
    backbonetuplep = BackboneFunction(molp, linf)

    # x3_Ntg -> Each rigid ring along the shortest path across the chain backbone, which
    # is surrounded by
    # rigid rings and/or by hydrogen-bonding backbone groups on both sides, contributes +1
    # to x3_Ntg.
    # The same contribution of +1 is made whether neighboring rigid rings are distinct from the
    # ring of interest, or members of the same fused ring unit.

    # distinct rigid ring surrounded by rigid ring - rings in the midle of chain
    if (len(backboneringtuple) >= 3):
        tmpvector = np.zeros([100], dtype=np.int64)
        tmpvectorbefore = np.zeros([100], dtype=np.int64)
        tmpvectorafter = np.zeros([100], dtype=np.int64)
        for i in range(1,
                       len(backboneringtuple) -
                       1):  # rings in the midle of chain
            tmpvector = ringstuple[backboneringtuple[i]]
            tmpmin = np.amin(tmpvector)
            tmpmax = np.amax(tmpvector)
            tmpvectorbefore = ringstuple[backboneringtuple[i - 1]]
            tmpbeforemax = np.amax(tmpvectorbefore)
            tmpvectorafter = ringstuple[backboneringtuple[i + 1]]
            tmpaftermin = np.amin(tmpvectorafter)
            if ((tmpbeforemax + 1) == (tmpmin)
                    and (tmpaftermin - 1) == tmpmax):
                x3_Ntg = x3_Ntg + 1

    # distinct rigid ring surrounded by rigid ring - rings at the ends of the chain
    if (len(backboneringtuple) == 1):  # one ring
        tmpvector = np.zeros([100], dtype=np.int64)
        tmpvector = ringstuple[backboneringtuple[0]]
        tmpmin = np.amin(tmpvector)
        tmpmax = np.amax(tmpvector)
        if (beginatom == tmpmin and endatom == tmpmax):
            x3_Ntg = x3_Ntg + 1

    if (len(backboneringtuple) >=
            2):  # two or more rings at the chain backbone
        tmpvector = np.zeros([100], dtype=np.int64)
        tmpvectorbefore = np.zeros([100], dtype=np.int64)
        tmpvectorafter = np.zeros([100], dtype=np.int64)
        pp = len(backboneringtuple) - 1
        tmpvector = ringstuple[backboneringtuple[0]]  # first ring
        tmpmin = np.amin(tmpvector)
        tmpmax = np.amax(tmpvector)
        tmpvectorafter = ringstuple[backboneringtuple[1]]
        tmpaftermin = np.amin(tmpvectorafter)
        tmpvectorbefore = ringstuple[backboneringtuple[pp]]
        tmpbeforemax = np.amax(tmpvectorbefore)
        if (beginatom == tmpmin and (tmpaftermin - 1) == tmpmax):
            if endatom == tmpbeforemax:
                x3_Ntg = x3_Ntg + 1
        pp = len(backboneringtuple) - 1
        tmpvector = ringstuple[backboneringtuple[pp]]  # last ring
        tmpmin = np.amin(tmpvector)
        tmpmax = np.amax(tmpvector)
        tmpvectorbefore = ringstuple[backboneringtuple[pp - 1]]
        tmpbeforemax = np.amax(tmpvectorbefore)
        tmpvectorafter = ringstuple[backboneringtuple[0]]
        tmpaftermin = np.amin(tmpvectorafter)
        if ((tmpbeforemax + 1) == tmpmin and endatom == tmpmax):
            if beginatom == tmpaftermin:
                x3_Ntg = x3_Ntg + 1

    # fused rigid ring surrounded by fused rigid ring -> fused rings in the midle of the chain
    if (len(backboneringtuple) >= 3):
        tmpvector = np.zeros([100], dtype=np.int64)
        tmpvectorbefore = np.zeros([100], dtype=np.int64)
        tmpvectorafter = np.zeros([100], dtype=np.int64)
        for i in range(1, len(backboneringtuple) - 1):
            a = np.isin(fusedtupleindex, backboneringtuple[i])
            if np.any(a):
                for j in range(0, len(fusedgrouptupleindex)):
                    for k in range(0, len(fusedgrouptupleindex[j])):
                        p = fusedgrouptupleindex[j][k]
                        test = np.isin(p, backboneringtuple)
                        if np.any(test):
                            if (
                                    p == backboneringtuple[i] and k == 0
                            ):  # first fused ring  it was p == 1 and k == 0
                                tmpvectorbefore = ringstuple[backboneringtuple[
                                    i - 1]]
                                tmpbeforemax = np.amax(tmpvectorbefore)
                                tmpvector = ringstuple[backboneringtuple[
                                    i]]  # = ringstuple[p] = ringstuple[fusedgrouptupleindex[j][k]]
                                tmpmin = np.amin(tmpvector)
                                test1 = np.isin(fusedgrouptupleindex[j][k + 1],
                                                backboneringtuple)
                                kk = len(fusedgrouptupleindex[k]) - 1
                                if np.any(test1):
                                    if ((tmpbeforemax + 1) == tmpmin
                                            and k < kk):
                                        x3_Ntg = x3_Ntg + 1
                            if (p == backboneringtuple[i]
                                    and k > 0):  # it was (p == i and k > 0)
                                if (k < (len(fusedgrouptupleindex[j]) -
                                         1)):  # fused ring between fused rings
                                    condition = 0
                                    test1 = np.isin(
                                        fusedgrouptupleindex[j][k - 1],
                                        backboneringtuple)
                                    if not np.any(test1):
                                        condition = 1
                                    test1 = np.isin(fusedgrouptupleindex[j][k],
                                                    backboneringtuple)
                                    if not np.any(test1):
                                        condition = 1
                                    test1 = np.isin(
                                        fusedgrouptupleindex[j][k + 1],
                                        backboneringtuple)
                                    if not np.any(test1):
                                        condition = 1
                                    kk = len(fusedgrouptupleindex[j]) - 1
                                    if condition == 0:
                                        if k < kk:
                                            x3_Ntg = x3_Ntg + 1
                                # last fused ring
                                tmpvectorafter = ringstuple[backboneringtuple[
                                    i + 1]]
                                tmpaftermin = np.amin(tmpvectorafter)
                                tmpvector = ringstuple[backboneringtuple[
                                    i]]  # = ringstuple[p] = ringstuple[fusedgrouptupleindex[j][k]]
                                tmpmax = np.amax(tmpvector)
                                test1 = np.isin(fusedgrouptupleindex[j][k - 1],
                                                backboneringtuple)
                                kk = len(fusedgrouptupleindex[j]) - 1
                                if np.any(test1):
                                    if (k == kk
                                            and (tmpaftermin - 1) == tmpmax):
                                        x3_Ntg = x3_Ntg + 1

    # fused rigid ring surrounded by distinct or fused rigid ring -> fused rings at
    # the ends of the chain
    tmpvector = np.zeros([100], dtype=np.int64)
    tmpvectorbefore = np.zeros([100], dtype=np.int64)
    tmpvectorafter = np.zeros([100], dtype=np.int64)
    ii = 0
    for i in range(0, len(ringstuple)):
        # looking for the ring index with the highest atom index
        jj = np.amax(ringstuple[i])
        if jj > ii:
            ii = jj
            pp = i

    if (len(fusedgrouptupleindex) != 0 and len(backboneringtuple) != 0):
        if backboneringtuple[0] == fusedgrouptupleindex[0][0]:
            tmpvector = ringstuple[
                backboneringtuple[0]]  # first backbone ring (fused ring)
            tmpmin = np.amin(tmpvector)
            test1 = np.isin(pp, backboneringtuple)
            tmpvectorbefore = ringstuple[pp]
            tmpbeforemax = np.amax(tmpvectorbefore)
            test2 = np.isin(fusedgrouptupleindex[0], backboneringtuple)
            if (test1 and np.any(test2)):
                if (endatom == tmpbeforemax and beginatom == tmpmin):
                    x3_Ntg = x3_Ntg + 1
        for i in range(0, len(fusedgrouptupleindex)):
            for j in range(0, len(fusedgrouptupleindex[i])):
                test1 = np.isin(pp, backboneringtuple)
                if (fusedgrouptupleindex[i][j] == pp
                        and test1):  # last backbone (fused ring)
                    tmpvector = ringstuple[pp]
                    tmpmax = np.amax(tmpvector)
                    tmpvectorafter = ringstuple[backboneringtuple[0]]
                    tmpaftermin = np.amin(tmpvectorafter)
                    test2 = np.isin(fusedgrouptupleindex[i][j - 1],
                                    backboneringtuple)
                    if test2:
                        if (beginatom == tmpaftermin and endatom == tmpmax):
                            x3_Ntg = x3_Ntg + 1

    # defining the pattern of hydrogen-bonding backbone groups
    patt_hbonding = Chem.MolFromSmarts(
        '[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0]),$([O,S;H1;v2]-[!$(*=[O,N,P,S]'
        + ')]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]'
    )
    patt_tuple = mol.GetSubstructMatches(patt_hbonding)
    patt_tuple = np.intersect1d(patt_tuple, backbonetuple)

    # a rigid ring surrounded by a hydrogen-bonding backbone groups on both sides
    tmpvector = np.zeros([100], dtype=np.int64)
    tmpvectorbefore = np.zeros([100], dtype=np.int64)
    tmpvectorafter = np.zeros([100], dtype=np.int64)
    if (len(patt_tuple) > 0):
        for i in range(0, len(ringstuple)):
            tmpvector = ringstuple[i]
            test = np.isin(ringstuple[i], shortestpathtuple)
            if np.any(test):
                tmpmin = np.amin(tmpvector)
                tmpmax = np.amax(tmpvector)
                tmpvectorbefore = patt_tuple
                tmpvectorbefore = np.asarray(tmpvectorbefore)
                pp = (np.abs(tmpvectorbefore - tmpmin)).argmin()
                tmpbeforemax = tmpvectorbefore[pp]
                tmpvectorafter = patt_tuple
                tmpvectorafter = np.asarray(tmpvectorafter)
                pp = (np.abs(tmpvectorafter - tmpmax)).argmin()
                tmpaftermin = tmpvectorafter[pp]
                if ((tmpbeforemax + 1) == tmpmin
                        and tmpmax == (tmpaftermin - 1)):
                    x3_Ntg = x3_Ntg + 1

    # x3_Ntg -> Each rigid ring along the shortest path across the chain backbone, which is
    # surrounded by any
    # combination of ether (-O-), thioether (-S-), and/or methylene (-CH2-) moieties on both sides,
    # contributes -1 to x3_Ntg.

    # defining the pattern of ether -O-, thioether -S- and methylene -CH2-
    patt_generic = Chem.MolFromSmarts('[OX2,SX2,CH2]')
    patt_tuple = molp.GetSubstructMatches(patt_generic)
    patt_tuple = np.intersect1d(patt_tuple, backbonetuplep)

    tmpvector = np.zeros([100], dtype=np.int64)
    tmpvectorbefore = np.zeros([100], dtype=np.int64)
    tmpvectorafter = np.zeros([100], dtype=np.int64)
    if (len(patt_tuple) > 0):
        for i in range(0, len(ringstuplep)):
            tmpvector = ringstuplep[i]
            test = np.isin(ringstuplep[i], shortestpathtuplep)
            if np.any(test):
                tmpmin = np.amin(tmpvector)
                tmpmax = np.amax(tmpvector)
                tmpvectorbefore = patt_tuple
                tmpvectorbefore = np.asarray(tmpvectorbefore)
                pp = (np.abs(tmpvectorbefore - tmpmin)).argmin()
                tmpbeforemax = tmpvectorbefore[pp]
                tmpvectorafter = patt_tuple
                tmpvectorafter = np.asarray(tmpvectorafter)
                pp = (np.abs(tmpvectorafter - tmpmax)).argmin()
                tmpaftermin = tmpvectorafter[pp]
                if ((tmpbeforemax + 1) == tmpmin
                        and tmpmax == (tmpaftermin - 1)):
                    x3_Ntg = x3_Ntg - 1

    return x3_Ntg


def x4NtgFunction(mol, lin):
    #   x4_Ntg is one of thirteen structural parameters which establishes
    #   a correlation with glass transition temperature Tg

    x4_Ntg = 0
    rigidring = RigidRingFunction(mol)
    ringstuple = RingsFunction(mol)
    fusedtupleindex = FusedRingsFunction(mol)
    backboneringtuple = BackboneRingFunction(mol, lin)

    # Finding the shortest path
    beginatom = int(lin[2])
    endatom = int(lin[3])
    shortestpathtuple = Chem.rdmolops.GetShortestPath(mol, beginatom, endatom)

    # Finding the bonds between the atoms of molecule
    bondidxs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                for b in mol.GetBonds()]
    bondidxs.append((beginatom, endatom))

    # x4_Ntg -> 1)
    # If a rigid ring is "distinct", i.e., if it is not a part of fused ring unit, then the number '
    # of
    # non-hydrogen atoms or groups sticking out from it is counted in determining x4_Ntg, but only
    # if
    # (a) two or more atoms and/or groups are sticking out from this ring, and (b) atoms and/or
    # groups are sticking out along both of the "paths" traversing this ring across the backbone
    # 2) If a distinct rigid ring has only one non-hydrogen substituent, or if it has more than one
    # substituent all attached to ring atoms located on the same chain backbone path traversing the
    # ring, no contribution is made to x4_Ntg

    tmpvector = np.zeros([100], dtype=np.int64)
    tmpvectorup = np.zeros([100], dtype=np.int64)
    for i in range(0, len(rigidring)):
        pp = rigidring[i]
        tmpvector = ringstuple[pp]
        tmpvectorup = np.intersect1d(shortestpathtuple, tmpvector)
        sideatomidxtmp = []
        sideatomidx = []
        sideatomupdown = []
        # up = 1; down = 0
        for j in range(
                0,
                len(bondidxs)):  # looking for side atoms attached to the ring
            test1 = np.isin(tmpvector, bondidxs[j][0])
            test2 = np.isin(tmpvector, bondidxs[j][1])
            if (np.any(test1) or np.any(test2)):
                if not np.any(test1):
                    sideatomidxtmp.append(bondidxs[j][0])
                if not np.any(test2):
                    sideatomidxtmp.append(bondidxs[j][1])
        sideatomidx = np.setdiff1d(
            sideatomidxtmp,
            shortestpathtuple)  # side atoms attached to the ring i
        for j in range(
                0, len(sideatomidx)
        ):  # looking if side atoms are at the top side or not of shortest path
            for k in range(0, len(bondidxs)):
                test5 = np.isin(bondidxs[k], ringstuple[pp])
                if np.any(test5):
                    if (bondidxs[k][0] == sideatomidx[j]
                            or bondidxs[k][1] == sideatomidx[j]):
                        if bondidxs[k][0] != sideatomidx[j]:
                            kk = bondidxs[k][0]
                            test3 = np.isin(tmpvectorup, kk)
                            if np.any(test3):
                                sideatomupdown.append(1)
                            else:
                                sideatomupdown.append(0)
                        if bondidxs[k][1] != sideatomidx[j]:
                            kk = bondidxs[k][1]
                            test3 = np.isin(tmpvectorup, kk)
                            if np.any(test3):
                                sideatomupdown.append(1)
                            else:
                                sideatomupdown.append(0)
        test4 = np.isin(sideatomupdown, 1)
        if not np.all(test4):
            test4 = np.isin(sideatomupdown, 0)
        if (len(sideatomupdown) >= 2 and not np.all(test4)):
            x4_Ntg = x4_Ntg + len(sideatomupdown)

    # x4_Ntg ->
    # 3) Each non-hydrogen substituent sticking out of a rigid ring located in a fused ring unit
    # along
    # the chain backbone contributes +1 to x4_Ntg, regardless of the number and location of such
    # substituents on the ring. The contribution of the rigid rings in each fused ring unit along
    # the
    # chain backbone to x4_Ntg is therefore equal to the total number of non-hydrogen substituents
    # sticking out from them.
    # 4) If a non-rigid ring is fused to a rigid ring, each non-hydrogen substituent sticking out
    # of the
    # two atoms of the non-rigid ring connecting this ring to the rigid ring also contributes +1 to
    # x4_Ntg
    # (This fourth statement was not implemented due to the difficult to identify if a fused ring
    # is a rigid or non-rigid ring)

    tmpvector = np.zeros([100], dtype=np.int64)
    sideatomidx = []
    for i in range(0, len(fusedtupleindex)):
        pp = fusedtupleindex[i]
        test = np.isin(pp, backboneringtuple)
        if test:
            tmpvector = ringstuple[pp]
            sideatomidxtmp1 = []
            sideatomidxtmp2 = []
            sideatomidx = []
            for j in range(0, len(
                    bondidxs)):  # looking for side atoms attached to the ring
                test1 = np.isin(tmpvector, bondidxs[j][0])
                test2 = np.isin(tmpvector, bondidxs[j][1])
                if (np.any(test1) or np.any(test2)):
                    if not np.any(test1):
                        sideatomidxtmp1.append(bondidxs[j][0])
                    if not np.any(test2):
                        sideatomidxtmp1.append(bondidxs[j][1])
            sideatomidxtmp2 = np.setdiff1d(sideatomidxtmp1, shortestpathtuple)
            for j in range(0, len(fusedtupleindex)):
                sideatomidx = np.setdiff1d(sideatomidxtmp2, ringstuple[
                    fusedtupleindex[j]])  # side atoms attached to the ring i
                sideatomidxtmp2 = sideatomidx
            x4_Ntg = x4_Ntg + len(sideatomidx)

    return x4_Ntg


def x5NtgFunction(mol, lin):
    #   x5_Ntg is one of thirteen structural parameters which establishes
    #   a correlation with glass transition temperature Tg

    x5_Ntg = 0
    ringstuple = RingsFunction(mol)
    backbonetuple = BackboneFunction(mol, lin)
    backboneringtuple = BackboneRingFunction(mol, lin)

    # Finding the bonds between the atoms of molecule
    bondidxs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                for b in mol.GetBonds()]
    beginatom = int(lin[2])
    endatom = int(lin[3])
    bondidxs.append((beginatom, endatom))

    # x5_Ntg ->
    # 1) If there are no "floppy" backbone rings along the chain backbone, then x5_Ntg=NBBrot, where
    # NBBrot was defined in Section 4.C. For example, x5_Ntg=2 for poly(vinyl alcohol) (Figure 4.2)
    # and x5_Ntg=8 for Ultem (Figure 2.4).
    # 2) Localized torsional motions of relatively limited amplitude, such as those around single
    # bonds in floppy rings, can contribute significantly to the heat capacity, and thus were
    # partially counted among the degrees of freedom for NBBrot. On the other hand, the glass
    # transition requires larger-scale cooperative chain motions. Atoms "tied down" in a ring are
    # not free to participate in such larger-scale motions even if the ring is floppy. Consequently,
    # if there are "floppy" backbone rings along the chain backbone, bonds located in such rings do
    # not contribute to x5_Ntg. For example, NBBrot=5 but x5_Ntg=2 for poly(vinyl butyral)
    # (Figure 2.9),
    # since the six-membered backbone ring contributes 3 to NBBrot, but does not contribute to X5.
    # 3) Each single bond in the backbone contributes +1 to NBBrot, provided that this single bond
    # is not in a ring. x5_Ntg = NBBrot
    # NBBrot is the number of backbone (BB) rotational bonds
    #
    # 4) Multiple bonds (double and triple bonds), either in the backbone or in side groups,
    # do not contribute either to NBBrot or to NSGrot
    # NSGrot is the number of side groups (SG) rotational bonds. Side groups are groups which does
    # not belong to the backbone chain
    # 5) Bonds in "rigid" rings, either in the backbone or in side groups, do not contribute
    # either to
    # NBBrot or to NSGrot.
    # In summary we just need to count the number of single bonds which does not belong to any ring
    # at the chain backbone

    tmpvector1 = []
    # bonds which do not belong to any ring
    if len(ringstuple) >= 1:  # it was >= 2
        for i in range(0, len(bondidxs)):
            condition = 0
            for j in range(0, len(ringstuple)):
                test1 = np.isin(bondidxs[i], ringstuple[j])
                if np.all(test1):
                    condition = 1
            if condition == 0:
                tmpvector1.append(bondidxs[i])
        if len(tmpvector1) == 0:
            tmpvector1.append((beginatom, endatom))
    if len(ringstuple) == 0:
        tmpvector1 = bondidxs

    condition = 0
    if (len(backboneringtuple) == 0 and len(ringstuple) != 0):
        test1 = np.isin((beginatom, endatom), tmpvector1)
        if not np.all(test1):
            tmpvector1.append((beginatom, endatom))
            condition = 1

    test1 = np.isin((beginatom, endatom), tmpvector1)
    if (not np.all(test1) and condition == 0):
        tmpvector1.append((beginatom, endatom))

    tmpvector2 = []
    # bonds which do not belong to any ring and at same time belong to the backbone chain
    for i in range(0, len(tmpvector1)):
        test1 = np.isin(tmpvector1[i], backbonetuple)
        if np.all(test1):
            tmpvector2.append(tmpvector1[i])

    tmpvector3 = []
    # bonds wich do not belong to any ring; and belong to the backbone chain; and are single bonds
    for i in range(0, len(tmpvector2) - 1):
        bondtmp = mol.GetBondBetweenAtoms(tmpvector2[i][0], tmpvector2[i][1])
        nbondtype = bondtmp.GetBondTypeAsDouble()
        if nbondtype == 1:
            tmpvector3.append(tmpvector2[i])

    if len(tmpvector2) > 0:
        i = len(tmpvector2) - 1
        tmpvector3.append(tmpvector2[i])

    x5_Ntg = len(tmpvector3)

    return x5_Ntg


def x6NtgFunction(mol, lin):
    # x6_Ntg is one of thirteen structural parameters which establishes
    # a correlation with glass transition temperature Tg

    x6_Ntg = 0
    ringstuple = RingsFunction(mol)
    backbonetuple = BackboneFunction(mol, lin)
    sidegrouptuple = SideGroupFunction(mol, lin)

    # Finding the bonds between the atoms of molecule
    bondidxs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                for b in mol.GetBonds()]
    beginatom = int(lin[2])
    endatom = int(lin[3])
    bondidxs.append((beginatom, endatom))

    # x6_Ntg ->
    # 1) If there are no floppy rings in any side groups, and no silicon atoms in the chain
    # backbone, then X6_Ntg = NSGrot
    # NSGrot is the number of side groups (SG) rotational bonds. Side groups are groups
    # which does not belong to the backbone chain
    # 2) If there are floppy rings in the side groups, these rings do not contribute to x6 although
    # they contribute to NSGrot
    # 3) Each single bond in a side group, or connecting a side group to the backbone, contributes
    # +1 to NSGrot, provided that (a) the rotations of this bond change the coordinates of at least
    # one atom, and (b) this bond is not in a ring
    #
    # 4) Multiple bonds (double and triple bonds), either in the backbone or in side groups,
    # do not contribute either to NBBrot or to NSGrot
    # NSGrot is the number of side groups (SG) rotational bonds. Side groups are groups which
    # does not belong to the backbone chain
    #
    # 5) Bonds in "rigid" rings, either in the backbone or in side groups, do not contribute either
    # to NBBrot or to NSGrot.
    # In summary we just need to count the number of single bonds which does not belong to any ring
    # at the side groups and the rotations of this bond change the coordinates of at least one atom.

    tmpvector1 = []
    # bonds which do not belong to any ring
    if len(ringstuple) != 0:
        for i in range(0, len(bondidxs)):
            condition = 0
            for j in range(0, len(ringstuple)):
                test1 = np.isin(bondidxs[i], ringstuple[j])
                if np.all(test1):
                    condition = 1
            if condition == 0:
                tmpvector1.append(bondidxs[i])
    else:
        tmpvector1 = bondidxs

    tmpvector2 = []
    # bonds which do not belong to any ring and at same time belong to the side group chain
    for i in range(0, len(tmpvector1)):
        test1 = np.isin(tmpvector1[i], sidegrouptuple)
        if np.any(test1):
            tmpvector2.append(tmpvector1[i])

    tmpvector3 = []
    # bonds wich do not belong to any ring; and belong to the side group chain; and are single bonds
    for i in range(0, len(tmpvector2)):
        bondtmp = mol.GetBondBetweenAtoms(tmpvector2[i][0], tmpvector2[i][1])
        nbondtype = bondtmp.GetBondTypeAsDouble()
        if nbondtype == 1:
            tmpvector3.append(tmpvector2[i])

    if len(tmpvector3) >= 2:
        for i in range(0, len(tmpvector3)):
            condition = 0
            for j in range(0, len(tmpvector3)):
                if j > i:
                    if tmpvector3[i][1] == tmpvector3[j][0]:
                        ii = tmpvector3[i][0]
                        jj = tmpvector3[i][1]
                        kk = tmpvector3[j][1]
                        molt = mol
                        AllChem.EmbedMolecule(molt, randomSeed=42)
                        conf = molt.GetConformer(0)
                        angle_degrees = Chem.rdMolTransforms.GetAngleDeg(
                            conf, int(ii), int(jj), int(kk))
                        if (angle_degrees != 180 and condition == 0):
                            # the rotation of this bond change the atom coordinate
                            x6_Ntg = x6_Ntg + 1
                            condition = 1
                        atom = mol.GetAtomWithIdx(kk)
                        st = atom.GetSymbol()
                        nh = atom.GetImplicitValence()
                        if (st != 'F' and nh != 0):
                            if st != 'C':
                                x6_Ntg = x6_Ntg + 1
                            if st == 'C':
                                if nh == 3:
                                    x6_Ntg = x6_Ntg + 1
                        test1 = atom.IsInRing()
                        if test1:  # bond between side group and ring
                            x6_Ntg = x6_Ntg + 1
                    if tmpvector3[i][0] == tmpvector3[j][0]:
                        atom1 = mol.GetAtomWithIdx(tmpvector3[i][1])
                        st1 = atom1.GetSymbol()
                        nh1 = atom1.GetImplicitValence()
                        if (st1 != 'F' and nh1 != 0):
                            if st1 != 'C':
                                x6_Ntg = x6_Ntg + 1
                            if st1 == 'C':
                                if nh1 == 3:
                                    x6_Ntg = x6_Ntg + 1
                        atom2 = mol.GetAtomWithIdx(tmpvector3[j][1])
                        st2 = atom2.GetSymbol()
                        nh2 = atom2.GetImplicitValence()
                        if (st2 != 'F' and nh2 != 0):
                            if st2 != 'C':
                                x6_Ntg = x6_Ntg + 1
                            if st2 == 'C':
                                if nh2 == 3:
                                    x6_Ntg = x6_Ntg + 1
            test1 = np.isin(tmpvector3[i][0], sidegrouptuple)
            test2 = np.isin(tmpvector3[i][1], sidegrouptuple)
            if np.any(test1):
                atom = mol.GetAtomWithIdx(tmpvector3[i][0])
                st = atom.GetSymbol()
                nh = atom.GetImplicitValence()
                test3 = atom.IsInRing()
                if (st != 'F' and nh
                        == 0):  # bond between the chain backbone and a ring
                    if test3:
                        x6_Ntg = x6_Ntg + 1
            if np.any(test2):
                atom = mol.GetAtomWithIdx(tmpvector3[i][1])
                st = atom.GetSymbol()
                nh = atom.GetImplicitValence()
                test3 = atom.IsInRing()
                if (st != 'F' and nh
                        == 0):  # bond between the chain backbone and a ring
                    if test3:
                        x6_Ntg = x6_Ntg + 1

    if len(tmpvector3) == 1:
        test1 = np.isin(tmpvector3[0][0], sidegrouptuple)
        test2 = np.isin(tmpvector3[0][1], sidegrouptuple)
        if np.any(test1):
            atom = mol.GetAtomWithIdx(tmpvector3[0][0])
            st = atom.GetSymbol()
            nh = atom.GetImplicitValence()
            test3 = atom.IsInRing()
            if (st != 'F' and nh != 0):
                x6_Ntg = x6_Ntg + 1
            if (st != 'F'
                    and nh == 0):  # bond between the chain backbone and a ring
                if test3:
                    x6_Ntg = x6_Ntg + 1
        if np.any(test2):
            atom = mol.GetAtomWithIdx(tmpvector3[0][1])
            st = atom.GetSymbol()
            nh = atom.GetImplicitValence()
            test3 = atom.IsInRing()
            if (st != 'F' and nh != 0):
                x6_Ntg = x6_Ntg + 1
            if (st != 'F'
                    and nh == 0):  # bond between the chain backbone and a ring
                if test3:
                    x6_Ntg = x6_Ntg + 1

    # x6_Ntg ->
    # 6 (a) If a backbone silicon atom has an ether (-O-), thioether (-S-), methylene
    # (-CH2-), or non-hydrogen-bonding (i.e., non-amide, non-urea and non-urethane) -NH- moiety
    # on at least one side along the chain backbone, each non-hydrogen atom in the side groups
    # attached to
    # that silicon atom contributes +1 to x6_Ntg, regardless of the actual number of rotational
    # degrees of
    # freedom in those side groups. For example, x6=NSGrot=2 for poly(dimethyl siloxane)
    # (Figure 5.2), where NSGrot equals the number of non-hydrogen atoms in the two side
    # (methyl) groups. On the other hand, for poly[oxy(methylphenylsilylene)] (Figure 6.11),
    # whose pendant phenyl ring only slightly enhances the chain stiffness, NSGrot=2 but x6=7.
    #
    # 6 (b) If a backbone silicon atom does not have any of the types of neighboring moieties
    # enumerated in (a) adjacent to it on either side along the chain backbone, then the side groups
    # attached to it are treated like side groups attached to any other backbone atom, and only
    # their degrees of freedom are counted in determining x6_Ntg.

    patt_generic = Chem.MolFromSmarts('[OX2,SX2,CH2,nH1]')
    patt_tuple = mol.GetSubstructMatches(patt_generic)

    # looking for Silicon atoms at backbone
    silicontuple = []
    siliconbackbonetuple = []
    natom = mol.GetNumAtoms()
    for i in range(0, natom):
        atom = mol.GetAtomWithIdx(i)
        sttmp = atom.GetSymbol()
        if sttmp == 'Si':
            silicontuple.append(i)

    siliconbackbonetuple = np.intersect1d(silicontuple, backbonetuple)
    patt_tuple_sidegroup = []
    patt_tuple_sidegroup = np.intersect1d(patt_tuple, sidegrouptuple)

    tmpvector = []
    for i in range(0, len(siliconbackbonetuple)):
        for j in range(0, len(patt_tuple_sidegroup)):
            tmpvector.append(siliconbackbonetuple[i])
            tmpvector.append(patt_tuple_sidegroup[j])
            test1 = np.isin(
                tmpvector3, tmpvector
            )  # tmpvector3 contain all the side group single bonds which are not in a ring
            for k in range(len(test1)):
                if np.all(test1[k]):
                    x6_Ntg = x6_Ntg + 1

    return x6_Ntg


def x7NtgFunction(mol, lin):
    #   x7_Ntg is one of thirteen structural parameters which establishes
    #   a correlation with glass transition temperature Tg

    x7_Ntg = 0
    ringstuple = RingsFunction(mol)
    backbonetuple = BackboneFunction(mol, lin)
    sidegrouptuple = SideGroupFunction(mol, lin)

    # Finding the bonds between the atoms of molecule
    bondidxs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                for b in mol.GetBonds()]
    beginatom = int(lin[2])
    endatom = int(lin[3])
    bondidxs.append((beginatom, endatom))

    # x7_Ntg ->
    # A symmetrically disubstituted backbone C or Si contributes +1 to x7_Ntg, except for such
    # atoms which are:
    # (a) located in rings, or
    # (b) flanked on both sides by rigid backbone rings and/or backbone hydrogen bonding groups.

    # looking for C or Si atoms at backbone chain which (a) are not located in rings and (b) are not
    # flanked on both sides by rigid backbone rings and/or backbone hydrogen bonding groups

    tmpvector1 = []
    # C or Si atoms
    natom = mol.GetNumAtoms()
    for i in range(0, natom):
        atom = mol.GetAtomWithIdx(i)
        sttmp = atom.GetSymbol()
        if (sttmp == 'Si' or sttmp == 'C'):
            tmpvector1.append(i)

    tmpvector2 = []
    # C or Si atoms belong to the backbone
    tmpvector2 = np.intersect1d(tmpvector1, backbonetuple)

    tmpvector3 = []
    # C or Si atoms belong to the backbone and do not belong to any ring
    if len(ringstuple) == 0:
        tmpvector3 = tmpvector2
    else:
        for i in range(0, len(ringstuple)):
            tmpvector3 = np.setdiff1d(tmpvector2, ringstuple[i])
            tmpvector2 = tmpvector3

    # patt_hbonding = Chem.MolFromSmarts('[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0])
    # ,$([n;H1;+0]),$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;
    # !$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]')
    patt_hbonding = Chem.MolFromSmarts('[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([S;H1;+0]),$([n;H1;+0]),'
                                       + '$([S;H1;v2]-[!$(*=[O,N,P,S])]),$([S;H0;v2]),$([O,S;-]),'
                                       + '$([N;v3;!$(N-*=!@[O,N,P,S])]),$([nH0,o,s;+0])]')
    patt_tuple = mol.GetSubstructMatches(patt_hbonding)

    bondbackbone = []
    # bonds between Si and C atoms at the backbone
    bondsidegroup = []
    # bonds between Si and C atoms and side groups atoms
    for i in range(0, len(bondidxs)):
        test1 = np.isin(bondidxs[i], backbonetuple)
        test2 = np.isin(bondidxs[i], sidegrouptuple)
        if np.all(test1):
            bondbackbone.append(bondidxs[i])
        if np.any(test2):
            bondsidegroup.append(bondidxs[i])

    ringstuple1d = []
    # flatten the n-dimensional ringstuple array into a 1d array
    for i in range(0, len(ringstuple)):
        for j in range(0, len(ringstuple[i])):
            ringstuple1d.append(ringstuple[i][j])

    tmpvector4 = []
    # C or Si atoms belong to the backbone; and do not belong to any ring; and arenot flanked on
    # both sides by rigid rings and/or hydrogen bonding groups
    for i in range(0, len(tmpvector3)):
        neighbor = []
        condition = 0
        for j in range(0, len(bondbackbone)):
            if tmpvector3[i] == bondbackbone[j][0]:
                neighbor.append(bondbackbone[j][1])
            if tmpvector3[i] == bondbackbone[j][1]:
                neighbor.append(bondbackbone[j][0])
        test1 = np.isin(neighbor, ringstuple1d)
        test2 = np.isin(neighbor, patt_tuple)
        if np.all(test1):
            condition = 1
        if np.all(test2):
            condition = 1
        if (np.any(test1) and np.any(test2)):
            condition = 1
        if condition == 0:
            tmpvector4.append(tmpvector3[i])

    symmetryvector = list(
        Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))

    # looking for symmetrical non-hydrogen side groups atoms attached to Si or C backbone atoms
    for i in range(0, len(tmpvector4)):
        # this is the vector with all C and Si atoms that fulfill all the requirements
        tmpvector5 = []
        for j in range(
                0, len(bondsidegroup)
        ):  # bondsidegroups list all bonds at the side groups of the chain
            if tmpvector4[i] == bondsidegroup[j][0]:
                tmpvector5.append(bondsidegroup[j][1])
            if tmpvector4[i] == bondsidegroup[j][1]:
                tmpvector5.append(bondsidegroup[j][0])
        if len(tmpvector5) == 2:
            ii = tmpvector5[0]
            jj = tmpvector5[1]
            if symmetryvector[ii] == symmetryvector[jj]:
                x7_Ntg = x7_Ntg + 1

    return x7_Ntg


def x8NtgFunction(mol, lin):
    #   x8_Ntg is one of thirteen structural parameters which establishes
    #   a correlation with glass transition temperature Tg

    x8_Ntg = 0
    sidegrouptuple = SideGroupFunction(mol, lin)
    ringstuple = RingsFunction(mol)
    deltaf = DeltaFunction(mol, lin)
    deltavf = DeltavFunction(mol, lin)

    # x8_Ntg -> 1) Each tertiary carbon atom (delta=deltav=4) in a side group contributes +1
    # to x8_Ntg whether it is in a ring or not in a ring.

    # looking for C or Si atoms at side groups
    tmpvector1c = []
    # C atoms
    tmpvector1si = []
    # Si atoms
    natom = mol.GetNumAtoms()
    for i in range(0, natom):
        atom = mol.GetAtomWithIdx(i)
        sttmp = atom.GetSymbol()
        if sttmp == 'C':
            tmpvector1c.append(i)
        if sttmp == 'Si':
            tmpvector1si.append(i)

    tmpvector2c = []
    # C atoms belong to the sidegroup
    tmpvector2c = np.intersect1d(tmpvector1c, sidegrouptuple)
    tmpvector2si = []
    # Si atoms belong to the sidegroup
    tmpvector2si = np.intersect1d(tmpvector1si, sidegrouptuple)

    for i in range(0, len(tmpvector2c)):
        ii = tmpvector2c[i]
        if (deltaf[ii] == 4 and deltavf[ii] == 4):
            x8_Ntg = x8_Ntg + 1

    # x8_Ntg ->
    # 2) Each methylene unit (carbon atom with delta=deltav=2) in a side group contributes -1 to
    # x8_Ntg unless it is located in a ring.
    # 3) Methylene units in rings of side groups do not contribute to x8_Ntg.
    # 4) Terminal =CH2 groups (carbon has deltav=2 but delta=1) are not counted as methylene units.

    # looking for C atoms belong to the side chain and does not belong to a ring
    tmpvector3c = []
    # C atoms belong to sidegroup and does not belong to a ring
    ringstuple1d = []
    # flatten the n-dimensional ringstuple array into a 1d array
    for i in range(0, len(ringstuple)):
        for j in range(0, len(ringstuple[i])):
            ringstuple1d.append(ringstuple[i][j])
    tmpvector3c = np.setdiff1d(tmpvector2c, ringstuple1d)

    for i in range(0, len(tmpvector3c)):
        ii = tmpvector3c[i]
        if (deltaf[ii] == 2 and deltavf[ii] == 2):
            x8_Ntg = x8_Ntg - 1

    # x8_Ntg ->
    # 5) Each tertiary silicon atom in a side group contributes +1 to x8_Ntg.
    for i in range(0, len(tmpvector2si)):
        ii = tmpvector2si[i]
        if (deltaf[ii] == 3 and deltavf[ii] == 1 / 3):
            x8_Ntg = x8_Ntg + 1

    return x8_Ntg


def x9NtgFunction(mol, lin):
    #   x9_Ntg is one of thirteen structural parameters which establishes
    #   a correlation with glass transition temperature Tg

    x9_Ntg = 0
    backbonetuple = BackboneFunction(mol, lin)
    sidegrouptuple = SideGroupFunction(mol, lin)

    # Finding the bonds between the atoms of molecule
    bondidxs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                for b in mol.GetBonds()]
    beginatom = int(lin[2])
    endatom = int(lin[3])
    bondidxs.append((beginatom, endatom))

    # x9_Ntg -> A asymmetrically disubstituted backbone C (carbon atom) contributes +1 to x9_Ntg

    # looking for C atoms at backbone chain
    tmpvector1 = []
    # C atoms
    natom = mol.GetNumAtoms()
    for i in range(0, natom):
        atom = mol.GetAtomWithIdx(i)
        sttmp = atom.GetSymbol()
        if sttmp == 'C':
            tmpvector1.append(i)

    tmpvector2 = []
    # C atoms belong to the backbone
    tmpvector2 = np.intersect1d(tmpvector1, backbonetuple)

    bondsidegroup = []
    # bonds between C atoms and side groups atoms
    for i in range(0, len(bondidxs)):
        test1 = np.isin(bondidxs[i], sidegrouptuple)
        if np.any(test1):
            bondsidegroup.append(bondidxs[i])

    symmetryvector = list(
        Chem.rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))

    # looking for asymmetrical non-hydrogen side groups atoms attached to C backbone atoms
    for i in range(0, len(tmpvector2)):
        # this is the vector with C atoms that fulfill all the requirements
        tmpvector3 = []
        for j in range(0, len(bondsidegroup)):
            # bondsidegroups list all bonds at the side groups of the chain
            if tmpvector2[i] == bondsidegroup[j][0]:
                tmpvector3.append(bondsidegroup[j][1])
            if tmpvector2[i] == bondsidegroup[j][1]:
                tmpvector3.append(bondsidegroup[j][0])
        if len(tmpvector3) == 2:
            ii = tmpvector3[0]
            jj = tmpvector3[1]
            if symmetryvector[ii] != symmetryvector[jj]:
                x9_Ntg = x9_Ntg + 1
    return x9_Ntg


def x10NtgFunction(mol, lin):
    #   x10_Ntg is one of thirteen structural parameters which establishes
    #   a correlation with glass transition temperature Tg

    x10_Ntg = 0
    ringstuple = RingsFunction(mol)
    fusedgrouptupleindex = GroupFusedRingsFunction(mol)
    backbonetuple = BackboneFunction(mol, lin)
    sidegrouptuple = SideGroupFunction(mol, lin)

    # Finding the bonds between the atoms of molecule
    bondidxs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                for b in mol.GetBonds()]
    beginatom = int(lin[2])
    endatom = int(lin[3])
    bondidxs.append((beginatom, endatom))

    # x10_Ntg ->
    # When a backbone carbon atom is attached by two bonds to a multiple-ring unit, all of whose
    # other atoms are in a side group, overcrowding of the atoms of the side group occurs in the
    # immediate vicinity of the chain backbone, resulting in increased chain stiffness and increased
    # Tg. The structural parameter x10_Ntg takes this effect into account. The contribution of each
    # such
    # multiple-ring moiety to x10_Ntg is equal to the number of non-hydrogen side group atoms in it.
    # looking for Carbon atoms at the polymer backbone
    tmpvector1 = []
    # C atoms
    natom = mol.GetNumAtoms()
    for i in range(0, natom):
        atom = mol.GetAtomWithIdx(i)
        sttmp = atom.GetSymbol()
        if sttmp == 'C':
            tmpvector1.append(i)

    tmpvector2 = []
    # C atoms belong to the backbone
    tmpvector2 = np.intersect1d(tmpvector1, backbonetuple)

    # looking for bonds between Carbon backbone atoms and side groups atoms
    bondsidegroup = []
    # bonds between side groups atoms
    for i in range(0, len(bondidxs)):
        test1 = np.isin(bondidxs[i], sidegrouptuple)
        if np.any(test1):
            bondsidegroup.append(bondidxs[i])

    bondcarbonsidegroup = []
    # bonds between Carbon backbone atoms and side groups atoms
    for i in range(0, len(bondsidegroup)):
        test1 = np.isin(bondsidegroup[i], tmpvector2)
        if np.any(test1):
            bondcarbonsidegroup.append(bondsidegroup[i])

    # looking if backbone Carbon atoms are attached by two bonds to a multiple-ring unit
    for i in range(
            0,
            len(tmpvector2)):  # this is the vector with Carbon backbone atoms
        tmpvector3 = []
        ii = 0
        for j in range(
                0, len(bondsidegroup)
        ):  # bondsidegroups list all bonds at the side groups of the chain
            if tmpvector2[i] == bondsidegroup[j][0]:
                tmpvector3.append(bondsidegroup[j][1])
            if tmpvector2[i] == bondsidegroup[j][1]:
                tmpvector3.append(bondsidegroup[j][0])
            if (len(tmpvector3) == 2 and ii == 0):
                # carbon backbone are attached to two side group atoms
                ii = 1
                tmpvector4 = []
                # this vector storage the ring indexes bonded to the carbon backbone atom from
                # tmpvector2
                for k in range(0, len(ringstuple)):
                    # looking if tmpvector3 atoms belong to fused rings
                    test1 = np.isin(tmpvector3, ringstuple[k])
                    if np.any(test1):
                        tmpvector4.append(k)
                tmpvector5 = []
                # this vector storage the fused ring atom indexes bonded
                # to the carbon backbone atom from tmpvector2
                for k in range(0, len(fusedgrouptupleindex)):
                    test2 = np.isin(tmpvector4, fusedgrouptupleindex[k])
                    if np.all(test2):
                        for m in range(0, len(tmpvector4)):
                            mm = tmpvector4[m]
                            tmpvector5.append(ringstuple[mm])
                tmpvector5_1d = []
                for k in range(0, len(tmpvector5)):
                    # flatten the tmpvector5 to 1d
                    for m in range(0, len(tmpvector5[k])):
                        tmpvector5_1d.append(tmpvector5[k][m])
                tmpvector6 = []
                # this vector storage all side group fused ring atoms bonded with two
                # bonds to carbon backbone atom
                for k in range(0, len(bondsidegroup)):
                    # looking for side group atoms attached to the fused ring
                    for m in range(0, len(tmpvector5_1d)):
                        if tmpvector5_1d[m] == bondsidegroup[k][0]:
                            test3 = np.isin(bondsidegroup[k][1], tmpvector5_1d)
                            if not np.any(test3):
                                tmpvector5_1d.append(bondsidegroup[k][1])
                        if tmpvector5_1d[m] == bondsidegroup[k][1]:
                            test3 = np.isin(bondsidegroup[k][0], tmpvector5_1d)
                            if not np.any(test3):
                                tmpvector5_1d.append(bondsidegroup[k][0])
                tmpvector6 = np.setdiff1d(tmpvector5_1d, tmpvector2)
                x10_Ntg = x10_Ntg + len(tmpvector6)

    return x10_Ntg


def x11NtgFunction(mol, lin):
    #   x11_Ntg is one of thirteen structural parameters which establishes
    #   a correlation with glass transition temperature Tg

    x11_Ntg = 0
    backbonetuple = BackboneFunction(mol, lin)

    # x11_Ntg -> x11_Ntg is the number of such C=C bonds located on the chain backbone
    #         with the double bond itself not located inside a ring

    nbond = mol.GetNumBonds()
    for i in range(0, nbond):
        bi = mol.GetBondWithIdx(i).GetBeginAtomIdx()
        atomi = mol.GetAtomWithIdx(bi)
        zi = atomi.GetAtomicNum()
        bf = mol.GetBondWithIdx(i).GetEndAtomIdx()
        atomf = mol.GetAtomWithIdx(bf)
        zf = atomf.GetAtomicNum()
        nbondtype = mol.GetBondWithIdx(i).GetBondTypeAsDouble()
        if (zi == 6 and zf == 6):
            if nbondtype == 2:
                if not atomi.IsInRing():
                    if not atomf.IsInRing():
                        test1 = np.isin(bi, backbonetuple)
                        test2 = np.isin(bf, backbonetuple)
                        if (np.any(test1) and np.any(test2)):
                            x11_Ntg = x11_Ntg + 1

    return x11_Ntg


def x12NtgFunction(mol, lin):
    # x12_Ntg is one of thirteen structural parameters which establishes
    # a correlation with glass transition temperature Tg

    x12_Ntg = 0
    backbonetuple = BackboneFunction(mol, lin)
    sidegrouptuple = SideGroupFunction(mol, lin)

    # Finding the bonds between the atoms of molecule
    bondidxs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                for b in mol.GetBonds()]
    beginatom = int(lin[2])
    endatom = int(lin[3])
    bondidxs.append((beginatom, endatom))

    # x12_Ntg -> x12_Ntg is defined as the number of fluorine atoms attached to
    #         non-aromatic atoms in side groups, where the backbone atom
    #         to which the side group itself is attached is not a silicon atom

    patt_aromatic = Chem.MolFromSmarts('a')
    patt_tuple = mol.GetSubstructMatches(patt_aromatic)

    nonSibackbone = []
    # looking for backbone atoms which is not a silicon atom
    natom = mol.GetNumAtoms()
    for i in range(0, natom):
        atom = mol.GetAtomWithIdx(i)
        sttmp = atom.GetSymbol()
        if sttmp != 'Si':
            nonSibackbone.append(i)

    nonSibackbone = np.intersect1d(nonSibackbone, backbonetuple)

    bondsidegroup = []
    # bonds between side groups atoms
    for i in range(0, len(bondidxs)):
        test1 = np.isin(bondidxs[i], sidegrouptuple)
        if np.any(test1):
            bondsidegroup.append(bondidxs[i])

    tmpvector1 = []
    # this vector storage side group atom indexes, where the backbone atom to which
    # the side group itself is attached is not a silicon atom
    for i in range(0, len(nonSibackbone)):
        for j in range(0, len(bondsidegroup)):
            if nonSibackbone[i] == bondsidegroup[j][0]:
                tmpvector1.append(bondsidegroup[j][1])
            if nonSibackbone[i] == bondsidegroup[j][1]:
                tmpvector1.append(bondsidegroup[j][0])

    tmpvector1 = np.setdiff1d(tmpvector1, backbonetuple)
    n1 = len(tmpvector1)
    n2 = 1000
    while (n1 != n2):
        n1 = len(tmpvector1)
        tmpvector1 = tmpvector1.tolist()
        for i in range(0, len(tmpvector1)):
            for j in range(0, len(bondsidegroup)):
                if tmpvector1[i] == bondsidegroup[j][0]:
                    tmpvector1.append(bondsidegroup[j][1])
                if tmpvector1[i] == bondsidegroup[j][1]:
                    tmpvector1.append(bondsidegroup[j][0])

        tmpvector1 = np.setdiff1d(tmpvector1, backbonetuple)
        n2 = len(tmpvector1)

    tmpvector2 = []
    # this vector storage flurine side group atom indexes, where the backbone atom
    # to which the side group itself is attached is not a silicon atom
    for i in range(0, len(tmpvector1)):
        atom = mol.GetAtomWithIdx(int(tmpvector1[i]))
        sttmp = atom.GetSymbol()
        if sttmp == 'F':
            tmpvector2.append(tmpvector1[i])

    tmpvector3 = []
    # this vector storage fluorine attached to non-aromatic side groups, where the backbone atom
    # to which the side group itself is attached is not a silicon atom
    for i in range(0, len(tmpvector2)):
        for j in range(0, len(bondsidegroup)):
            if tmpvector2[i] == bondsidegroup[j][0]:
                test1 = np.isin(bondsidegroup[j][1], patt_tuple)
                if not np.any(test1):
                    tmpvector3.append(tmpvector2[i])
            if tmpvector2[i] == bondsidegroup[j][1]:
                test1 = np.isin(bondsidegroup[j][0], patt_tuple)
                if not np.any(test1):
                    tmpvector3.append(tmpvector2[i])

    tmpvector4 = []
    # eliminating side group fluorine atoms attached to the backbone
    for i in range(0, len(tmpvector3)):
        for j in range(0, len(bondsidegroup)):
            if tmpvector3[i] == bondsidegroup[j][0]:
                test1 = np.isin(bondsidegroup[j][1], backbonetuple)
                if not np.any(test1):
                    tmpvector4.append(tmpvector3[i])
            if tmpvector3[i] == bondsidegroup[j][1]:
                test1 = np.isin(bondsidegroup[j][0], backbonetuple)
                if not np.any(test1):
                    tmpvector4.append(tmpvector3[i])

    x12_Ntg = len(tmpvector4)

    return x12_Ntg


def x13NtgFunction(mol, lin):
    #   x13_Ntg is one of thirteen structural parameters which establishes
    #   a correlation with glass transition temperature Tg

    x13_Ntg = 0
    backbonetuple = BackboneFunction(mol, lin)
    sidegrouptuple = SideGroupFunction(mol, lin)
    Nbb = len(backbonetuple)  # Nbb is the number of chain backbone atoms

    # Finding the bonds between the atoms of molecule
    bondidxs = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                for b in mol.GetBonds()]
    beginatom = int(lin[2])
    endatom = int(lin[3])
    bondidxs.append((endatom, beginatom))

    # defining the molecule with periodic bonds to calculate fragments
    molp = molextFunction(mol, lin)
    # print(Chem.MolToSmiles(molp))

    # x13_Ntg ->
    # The value of x13_Ntg equals the total number of non-hydrogen side
    # group atoms attached to the backbone silicon atoms flanked by ether (-O-) linkages on both
    # sides along the chain backbone, divided by the number NBB of atoms on the chain backbone.
    # The structural parameter x13_Ntg calculated by using this definition is automatically an
    # intensive quantity, with NBB rather than N (total number of atoms in the hydrogen-suppressed
    # graph) as the scaling factor

    # looking for Si atom at the backbone chain
    tmpvector1 = []
    # this vector storage the chain backbone Si atoms
    natom = mol.GetNumAtoms()
    for i in range(0, natom):
        atom = mol.GetAtomWithIdx(i)
        sttmp = atom.GetSymbol()
        if sttmp == 'Si':
            tmpvector1.append(i)

    tmpvector1 = np.intersect1d(tmpvector1, backbonetuple)

    bondbackbone = []
    # bonds between atoms at the backbone
    bondsidegroup = []
    # bonds between side groups atoms
    for i in range(0, len(bondidxs)):
        test1 = np.isin(bondidxs[i], backbonetuple)
        test2 = np.isin(bondidxs[i], sidegrouptuple)
        if np.all(test1):
            bondbackbone.append(bondidxs[i])
        if np.any(test2):
            bondsidegroup.append(bondidxs[i])

    patt_ether = Chem.MolFromSmarts('[O;D2]')
    patt_tuple = molp.GetSubstructMatches(patt_ether)
    patt_array = np.array(patt_tuple)

    for i in range(0, len(patt_array)):
        patt_array[i] = patt_array[i] - 1
    # print('patt_array - ether (-O-)',patt_array)

    # looking for Si atoms at the backbone chain which are flanked by ether (-O-)
    # linkages on both sides along the chain backbone
    tmpvector2 = []
    # this vector storage Si atoms belong to the backbone; which are
    # flanked onboth sides ether (-O-) linkages along the chain backbone
    for i in range(0, len(tmpvector1)):
        neighbor = []
        for j in range(0, len(bondbackbone)):
            if tmpvector1[i] == bondbackbone[j][0]:
                neighbor.append(bondbackbone[j][1])
            if tmpvector1[i] == bondbackbone[j][1]:
                neighbor.append(bondbackbone[j][0])
        # print('neighbor',tmpvector1[i],neighbor)
        test1 = np.isin(neighbor, patt_array)
        # print('is in ether pattern tuple',test1)
        if np.all(test1):
            tmpvector2.append(tmpvector1[i])

    # print('tmpvector2 Si flanked by -O- in both sides',tmpvector2)

    tmpvector3 = []
    # this vector storage side group atom indexes attached to Si atoms at the backbone
    # chain which are flanked by ether (-O-) linkages
    for i in range(0, len(tmpvector2)):
        for j in range(0, len(bondsidegroup)):
            if tmpvector2[i] == bondsidegroup[j][0]:
                tmpvector3.append(bondsidegroup[j][1])
            if tmpvector2[i] == bondsidegroup[j][1]:
                tmpvector3.append(bondsidegroup[j][0])

    tmpvector3 = np.setdiff1d(tmpvector3, backbonetuple)
    n1 = len(tmpvector3)
    n2 = 1000
    while (n1 != n2):
        n1 = len(tmpvector3)
        tmpvector3 = tmpvector3.tolist()
        for i in range(0, len(tmpvector3)):
            for j in range(0, len(bondsidegroup)):
                if tmpvector3[i] == bondsidegroup[j][0]:
                    tmpvector3.append(bondsidegroup[j][1])
                if tmpvector3[i] == bondsidegroup[j][1]:
                    tmpvector3.append(bondsidegroup[j][0])
        tmpvector3 = np.setdiff1d(tmpvector3, backbonetuple)
        n2 = len(tmpvector3)

    x13_Ntg = len(tmpvector3) / Nbb

    return x13_Ntg


def waterdensityFunction(temperature, pressure):
    # This function calculates the water density as a function of
    # temperature and pressure
    #
    # input:  temperature in Kelvin
    #         pressure in Pa
    #
    # output: rho_f in Kg/m^3
    #
    # This function was implemented accord to the following reference:
    #
    # J. Phys. Chem. Ref. Data, Vol. 31, No. 2, 2002
    # The IAPWS Formulation 1995 for the Thermodynamic Properties
    # of Ordinary Water Substance for General and Scientific Use
    #
    # ---------------------------------------------------------------------
    # some constants
    T_c = 647.096  # in Kelvin K
    p_c = 22.064  # MPa
    rho_c = 322  # Kg/m^3

    # vapor pressure
    T = temperature
    alfa = (1 - T / T_c)
    a1 = -7.85951783
    a2 = 1.84408259
    a3 = -11.7866497
    a4 = 22.6807411
    a5 = -15.9618719
    a6 = 1.80122502
    a = (T_c / T) * (a1 * alfa + a2 * alfa**1.5 + a3 * alfa**3 +
                     a4 * alfa**3.5 + a5 * alfa**4 + a6 * alfa**7.5)
    p_sigma = p_c * math.exp(a)  # p_sigma is in MPa

    # saturated liquid density
    b1 = 1.99274064
    b2 = 1.09965342
    b3 = -0.510839303
    b4 = -1.75493479
    b5 = -45.5170352
    b6 = -6.74694450e5
    rho_liq = rho_c * (1 + b1 * alfa**(1 / 3) + b2 * alfa**(2 / 3) +
                       b3 * alfa**(5 / 3) + b4 * alfa**(16 / 3) +
                       b5 * alfa**(43 / 3) + b6 * alfa**(110 / 3))

    # saturated vapor density
    c1 = -2.03150240
    c2 = -2.68302940
    c3 = -5.38626492
    c4 = -17.2991605
    c5 = -44.7586581
    c6 = -63.9201063
    c = c1 * alfa**(2 / 6) + c2 * alfa**(4 / 6) + c3 * alfa**(
        8 / 6) + c4 * alfa**(18 / 6) + c5 * alfa**(37 / 6) + c6 * alfa**(71 /
                                                                         6)
    rho_vapor = rho_c * math.exp(c)

    if pressure > p_sigma * 1e6:  # conversion of p_sigma from MPa to Pa
        rho_f = rho_liq
    else:
        rho_f = rho_vapor

    return rho_f


def waterviscosityFunction(temperature, pressure):
    #   This function calculates the water viscosity (in Pa.s) as a function of
    #   temperature for any pressure and temperature away from triple point.
    #   This function was implemented following the references:
    #
    #   J. Phys. Chem. Ref. Data, Vol. 38, No. 2, 2009
    #                  and
    #   IAPWS R12-08 The International Association for the Properties of Water and Steam
    #

    rho = waterdensityFunction(temperature, pressure)
    ns = 0  # initialization of variable water viscosity in Pa.s
    t = temperature  # temperature in Kelvin
    # if t <= 373.16:
    # rho = 997  # water density is 997 Kg/m^3
    # else:
    # rho = 0.4  # water density is 0.4 Kg/m^3
    ns_r = 1e-6  # reference water viscosity in Pa.s
    rho_r = 322.0  # reference water density in Kg/m^3
    t_r = 647.096  # reference temperature in Kelvin
    rho_ad = rho / rho_r
    t_ad = t / t_r

    h0 = [1.67752, 2.20462, 0.6366564, -0.241605]
    sumt = 0
    for i in range(0, 4):
        sumt = sumt + h0[i] / ((t_ad)**i)

    ns0_ad = 100 * math.sqrt(t_ad) / (sumt)

    h1 = [[
        5.20094e-1, 2.22531e-1, -2.81378e-1, 1.61913e-1, -3.25372e-2, 0.000,
        0.000
    ], [8.50895e-2, 9.99115e-1, -9.06851e-1, 2.57399e-1, 0.000, 0.000, 0.000],
        [-1.08374, 1.88797, -7.72479e-1, 0.000, 0.000, 0.000, 0.000],
        [
        -2.89555e-1, 1.26613, -4.89837e-1, 0.000, 6.98452e-2, 0.000,
        -4.35673e-3
    ], [0.000, 0.000, -2.57040e-1, 0.000, 0.000, 8.72102e-3, 0.000],
        [0.000, 1.20573e-1, 0.000, 0.000, 0.000, 0.000, -5.93264e-4]]

    sumt1 = 0
    for i in range(0, 6):
        for j in range(0, 7):
            sumt1 = sumt1 + (((1 / t_ad) - 1)**i) * (h1[i][j] *
                                                     (rho_ad - 1)**j)

    ns1_ad = math.exp(rho_ad * sumt1)

    ns_ad = ns0_ad * ns1_ad
    ns = ns_ad * ns_r  # in Pa.s

    return ns

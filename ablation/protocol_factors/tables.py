"""Tabulated reference values (25 C unless noted). Sources: CRC Handbook of Chemistry and
Physics (solvent dielectric constants, boiling and melting points); Good et al., Biochemistry
5:467 (1966) and standard biological buffer tables (buffer pKa, dpKa/dT); common aqueous pKa
tables for simple acids/bases. Values not listed are reported as missing, never guessed."""
# name: (dielectric, bp_C, mp_C, protic)
SOLVENTS = {
 "water": (78.4, 100.0, 0.0, True), "methanol": (32.7, 64.7, -97.6, True),
 "ethanol": (24.5, 78.4, -114.1, True), "isopropanol": (19.9, 82.5, -89.0, True),
 "acetone": (20.7, 56.1, -94.7, False), "acetonitrile": (37.5, 81.6, -45.7, False),
 "dimethyl sulfoxide": (46.7, 189.0, 18.5, False), "dimethylformamide": (36.7, 153.0, -61.0, False),
 "tetrahydrofuran": (7.6, 66.0, -108.4, False), "diethyl ether": (4.3, 34.6, -116.3, False),
 "dichloromethane": (8.9, 39.6, -96.7, False), "chloroform": (4.8, 61.2, -63.5, False),
 "toluene": (2.4, 110.6, -95.0, False), "hexane": (1.9, 68.7, -95.0, False),
 "ethyl acetate": (6.0, 77.1, -83.6, False), "ethylene carbonate": (89.8, 248.0, 36.4, False),
 "propylene carbonate": (64.9, 242.0, -48.8, False), "dimethyl carbonate": (3.1, 90.0, 4.6, False),
 "diethyl carbonate": (2.8, 126.0, -43.0, False), "1,2-dimethoxyethane": (7.2, 85.0, -58.0, False),
 "acetic acid": (6.2, 118.0, 16.6, True), "pyridine": (12.4, 115.0, -41.6, False),
 "benzene": (2.3, 80.1, 5.5, False), "1,4-dioxane": (2.2, 101.0, 11.8, False),
 "glycerol": (42.5, 290.0, 17.8, True), "formamide": (111.0, 210.0, 2.6, True),
 "n-methyl-2-pyrrolidone": (32.2, 202.0, -24.0, False), "heptane": (1.9, 98.4, -90.6, False),
 "2-propanol": (19.9, 82.5, -89.0, True), "1-butanol": (17.5, 117.7, -89.8, True),
}
SOLVENT_SMILES = {
 "water": "O", "methanol": "CO", "ethanol": "CCO", "isopropanol": "CC(C)O", "2-propanol": "CC(C)O",
 "acetone": "CC(C)=O", "acetonitrile": "CC#N", "dimethyl sulfoxide": "CS(C)=O",
 "dimethylformamide": "CN(C)C=O", "tetrahydrofuran": "C1CCOC1", "diethyl ether": "CCOCC",
 "dichloromethane": "ClCCl", "chloroform": "ClC(Cl)Cl", "toluene": "Cc1ccccc1", "hexane": "CCCCCC",
 "ethyl acetate": "CCOC(C)=O", "ethylene carbonate": "O=C1OCCO1", "propylene carbonate": "CC1COC(=O)O1",
 "dimethyl carbonate": "COC(=O)OC", "diethyl carbonate": "CCOC(=O)OCC", "1,2-dimethoxyethane": "COCCOC",
 "acetic acid": "CC(=O)O", "pyridine": "c1ccncc1", "benzene": "c1ccccc1", "1,4-dioxane": "C1COCCO1",
 "glycerol": "OCC(O)CO", "formamide": "NC=O", "n-methyl-2-pyrrolidone": "CN1CCCC1=O",
 "heptane": "CCCCCCC", "1-butanol": "CCCCO",
}
# name: (smiles of neutral/parent form, [pKa values], dpKa/dT per K or None)
PKA = {
 "acetic acid": ("CC(=O)O", [4.76], None), "formic acid": ("OC=O", [3.75], None),
 "benzoic acid": ("OC(=O)c1ccccc1", [4.20], None),
 "phosphoric acid": ("OP(=O)(O)O", [2.15, 7.20, 12.35], -0.0028),
 "carbonic acid": ("OC(=O)O", [6.35, 10.33], None), "ammonia": ("N", [9.25], None),
 "tris": ("NC(CO)(CO)CO", [8.07], -0.028), "hepes": ("OCCN1CCN(CCS(=O)(=O)O)CC1", [7.48], -0.014),
 "mes": ("OS(=O)(=O)CCN1CCOCC1", [6.15], -0.011), "mops": ("OS(=O)(=O)CCCN1CCOCC1", [7.20], -0.015),
 "pipes": ("OS(=O)(=O)CCN1CCN(CCS(=O)(=O)O)CC1", [6.76], -0.0085),
 "bicine": ("OCCN(CCO)CC(=O)O", [8.26], -0.018), "tricine": ("OCC(CO)(CO)NCC(=O)O", [8.05], -0.021),
 "bis-tris": ("OCCN(CCO)C(CO)(CO)CO", [6.46], -0.017), "glycine": ("NCC(=O)O", [2.34, 9.60], None),
 "citric acid": ("OC(=O)CC(O)(CC(=O)O)C(=O)O", [3.13, 4.76, 6.40], None),
 "edta": ("OC(=O)CN(CC(=O)O)CCN(CC(=O)O)CC(=O)O", [2.0, 2.7, 6.16, 10.26], None),
 "imidazole": ("c1c[nH]cn1", [6.95], None), "phenol": ("Oc1ccccc1", [9.99], None),
 "boric acid": ("OB(O)O", [9.24], None), "triethylamine": ("CCN(CC)CC", [10.75], None),
 "pyridine": ("c1ccncc1", [5.25], None), "hydrofluoric acid": ("F", [3.17], None),
 "hypochlorous acid": ("OCl", [7.53], None), "hydrogen sulfide": ("S", [7.0], None),
 "ammonium": ("[NH4+]", [9.25], None),
}

# Battery-electrolyte solvents: (dielectric, viscosity cP, mp C, bp C); K. Xu, Chem. Rev. 104, 4303 (2004), Table 1
# (EC viscosity at 40 C; others at 25 C). Missing values are left out.
ELECTROLYTE_SOLVENTS = {
 "EC": (89.78, 1.90, 36.4, 248), "PC": (64.92, 2.53, -48.8, 242), "DMC": (3.107, 0.59, 4.6, 91),
 "DEC": (2.805, 0.75, -74.3, 126), "EMC": (2.958, 0.65, -53, 110), "DME": (7.2, 0.46, -58, 84),
 "DOL": (7.1, 0.59, -95, 78), "THF": (7.4, 0.46, -109, 66), "2-MeTHF": (6.2, 0.47, -137, 80),
 "2-Glyme": (7.4, 1.06, -64, 162), "3-Glyme": (7.5, 1.96, -46, 216), "4-Glyme": (7.9, 3.39, -30, 275),
 "g-Butyrolactone": (39.0, 1.73, -43.5, 204), "AN": (35.95, 0.341, -48.8, 81.6),
 "EA": (6.02, 0.45, -84, 77), "DMSO": (46.7, 1.99, 18.5, 189),
}

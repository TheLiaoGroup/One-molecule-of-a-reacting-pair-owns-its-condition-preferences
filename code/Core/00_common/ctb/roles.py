"""Condition variables: which reagent in each condition plays which role (catalyst precursor, ligand,
base, additive, solvent, ...), read from the reagent names in DATASET_conditions.csv. Name-based
corrections move reagents recorded in the wrong slot to their chemical role. Taken unchanged from the
condition-variable analysis that first defined these roles (see the provenance record)."""
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- roles
# slot -> variable name, per screen. Slots absent here are ignored.
_PD = {"Reagent1": "precatalyst", "Reagent2": "ligand", "Reagent3": "base", "Reagent4": "additive", "solvent": "solvent"}
ROLES = {
    "Acylation|1":                 {"Reagent1": "activator", "Reagent2": "additive", "Reagent3": "base", "solvent": "solvent"},
    "Buchwald-Hartwig(CMC)|1": _PD, "Buchwald-Hartwig(CMC)|2": _PD, "Buchwald-Hartwig(CMC)|3": _PD,
    "Buchwald-Hartwig|1": _PD, "Buchwald-Hartwig|2": _PD, "C-O competitive reactions|1": _PD, "Suzuki|1": _PD,
    "Heck|1": _PD, "Ullmann|1": _PD,
    "Chan-Lam|1":                  {"Reagent1": "precatalyst", "Reagent2": "ligand", "Reagent3": "base", "Reagent4": "oxidant", "solvent": "solvent"},
    "Esterification|1":            {"Reagent1": "promoter", "Reagent2": "additive", "Reagent3": "base", "Reagent4": "coupling reagent", "Reagent5": "additive", "solvent": "solvent"},
    "Grignard|1":                  {"Reagent1": "additive", "Reagent2": "additive", "solvent": "solvent"},
    "Hiyama|1":                    {"Reagent1": "precatalyst", "Reagent2": "ligand", "Reagent3": "base", "Reagent4": "fluoride", "solvent": "solvent"},
    "Kumada|1":                    {"Reagent1": "precatalyst", "Reagent2": "ligand", "Reagent3": "additive", "solvent": "solvent"},
    "Reductive-amination|1":       {"Reagent1": "Lewis acid", "Reagent3": "reductant", "Reagent4": "acid", "solvent": "solvent"},
    "SNAr|1":                      {"Reagent3": "base", "solvent": "solvent"},
    "Sonogashira|1":               {"Reagent1": "precatalyst", "Reagent2": "co-catalyst", "Reagent3": "ligand", "Reagent4": "base", "Reagent5": "additive", "solvent": "solvent"},
}
METAL_SCREENS = {s for s in ROLES if "precatalyst" in ROLES[s].values()}
# name-based corrections where a slot holds more than one role
KUMADA_SLOT2_ADDITIVES = {"TMEDA", "TEA", "NMP", "DMPU", "DMI", "TMU", "NBP", "N-Me-CPL"}        # amines and amide co-solvents listed in the ligand slot
ESTER_COUPLING = {"DCC", "EDC-HCl", "DIC", "CDI", "HATU", "TBTU", "BzCl", "(COCl)2"}              # acid-activating reagents, whichever slot lists them
LIGAND_SLOT_ADDITIVES = {"Bu4NBr", "CsF", "LiCl", "TBAI", "TBHQ", "sodium ascorbate", "EtOAc"}   # listed in the ligand slot of a metal-catalysed array but not ligands: salts, a radical inhibitor, a reductant, a co-solvent
BASE_SLOT_ADDITIVES = {"Zn(OAc)2", "Hydroquinone", "Cu(OAc)2", "TMEDA", "Bu4NBr", "AgTFA"}   # listed in a base slot but not bases:
# a zinc salt, a radical inhibitor, a copper oxidant, a diamine, a phase-transfer salt and a silver halide scavenger
GRIGNARD_CATALYST = {"CuI", "CuBr", "CuBr.Me2S", "Cu", "Pd(PPh3)4", "Iron dibzacac", "CoCl2", "Cp2TiCl2", "MTO", "1,10-Phen", "(2,2:6,2-Terpyridin)-4-ol"}   # metal sources and ligands in the Grignard additive slots
AMINE_SOLVENTS = ["TEA", "iPr2NH", "sBuNH2", "DIPEA", "NEt3", "Et3N", "piperidine", "pyrrolidine", "morpholine", "NHEt2", "Cy2NMe"]   # an amine used as (co-)solvent is the base when the base slot is empty

_METALS = ["Pd", "Ni", "Cu", "Fe", "Au", "Ag", "Ti", "Sn", "Zn", "Co", "Ce", "Cd", "In", "Mn", "Ir", "Rh", "Ru"]
def metal_of(name):
    n = str(name)
    if "nickel" in n.lower(): return "Ni"
    if "PEPPSI" in n or "palladium" in n.lower(): return "Pd"
    for m in _METALS:
        if re.search(r'(^|[^A-Za-z])' + m + r'(?![a-z])', n): return m
    return "none"

_NONLIG = {"oac", "acac", "mecn", "phcn", "dba", "cod", "allyl", "cinnamyl", "methylallyl", "otf", "tfa", "no3", "glyme",
           "toluene", "benzene", "chcl3", "ch2cl2", "h2o", "oh", "cl", "br", "i", "so4", "cl2", "br2", "2", "3", "4", "ii", "c",
           "et2o", "me2s", "5h2o", "6h2o"}
def norm_lig(s):
    s = str(s).lower().replace("\u00b7", ".")
    s = re.sub(r'\.(hbf4|hcl|chcl3|ch2cl2|toluene|benzene|me2s)$', '', s)
    s = re.sub(r'[\s\-\.]', '', s)
    s = s.replace("tbu3)", "tbu)3")
    return s

def embedded_ligand(name):
    """Ligand carried inside a precatalyst name; 'none' if the precatalyst has no phosphine/NHC/amine ligand."""
    n = str(name).strip()
    if n == "nan" or not n: return "none"
    if "nickel(II) chloride" in n: return "nnn-pincer"
    if n.startswith("trans-di("): return "herrmann"          # Herrmann's palladacycle, P(o-tolyl)3-derived
    m = re.match(r'^(.+?)[\s\-]+Pd[\s\-]*(G\d)?$', n)
    if m: return norm_lig(m.group(1))
    m = re.match(r'^PEPPSI-(.+)$', n)
    if m: return norm_lig(m.group(1))
    m = re.match(r'^\((.+?)\)Pd\(allyl\)Cl$', n)
    if m: return norm_lig(m.group(1))
    m = re.match(r'^\[Cu\(OH\)(.+?)\]', n)
    if m: return norm_lig(m.group(1))
    if n.startswith("[PdP(tBu)3Br]2"): return norm_lig("P(tBu)3")
    found = []
    for g in re.findall(r'[\(\[]([^\(\)\[\]]*(?:\([^\(\)]*\))*[^\(\)\[\]]*)[\)\]]', n):
        g2 = norm_lig(g)
        if g2 in _NONLIG or not g2: continue
        if re.match(r'^(pd|ni|cu|fe)\(', g2): continue
        if re.fullmatch(r'[a-z0-9]{1,2}', g2): continue
        found.append(g2)
    found = [f for f in found if f not in _NONLIG]
    return "+".join(sorted(set(found))) if found else "none"

def condition_variables(screen, C, rxn_class, cond_ids):
    """DataFrame indexed by condition_id: one column per variable (levels as strings; missing slot -> 'none')."""
    cc = C[(C.rxn_class == rxn_class) & (C.condition_id.isin(cond_ids))].set_index("condition_id").reindex(cond_ids)
    roles = ROLES[screen]; V = {}
    def merge(role, vals):
        if role in V: V[role] = [a if b == "none" else (b if a == "none" else a + "+" + b) for a, b in zip(V[role], vals)]
        else: V[role] = list(vals)
    for slot, role in roles.items():
        col = "Standardized Name" if slot == "solvent" else slot + "_Standardized Name"
        vals = list(cc[col].fillna("none").astype(str))
        if screen == "Kumada|1" and slot == "Reagent2":                      # the ligand slot also lists additives: split by name
            merge("additive", [v if v in KUMADA_SLOT2_ADDITIVES else "none" for v in vals]); vals = [v if v not in KUMADA_SLOT2_ADDITIVES else "none" for v in vals]
        if screen == "Esterification|1" and slot != "solvent":               # coupling reagents sit in slots 1, 2 and 4; other slot-4 entries are additives
            merge("coupling reagent", [v if v in ESTER_COUPLING else "none" for v in vals])
            vals = [v if v not in ESTER_COUPLING else "none" for v in vals]
            if role == "coupling reagent": role = "additive"
        if role == "base" and slot != "solvent":                               # a salt and a radical inhibitor listed in the base slot are additives
            merge("additive", [v if v in BASE_SLOT_ADDITIVES else "none" for v in vals]); vals = [v if v not in BASE_SLOT_ADDITIVES else "none" for v in vals]
        if role == "ligand" and slot != "solvent":                             # salts, a radical inhibitor, a reductant and a co-solvent listed in the ligand slot are additives
            merge("additive", [v if v in LIGAND_SLOT_ADDITIVES else "none" for v in vals]); vals = [v if v not in LIGAND_SLOT_ADDITIVES else "none" for v in vals]
        if screen == "Grignard|1" and slot != "solvent":                       # the two additive slots also hold metal sources and ligands: those are the catalyst variable
            merge("precatalyst", [v if v in GRIGNARD_CATALYST else "none" for v in vals]); vals = [v if v not in GRIGNARD_CATALYST else "none" for v in vals]
        if role == "ligand": vals = [norm_lig(v) for v in vals]
        merge(role, vals)
    if "precatalyst" in roles.values():
        pre = cc["Reagent1_Standardized Name"].astype(str)
        V["metal"] = list(pre.map(metal_of))
        emb = pre.map(embedded_ligand)
        add = V.get("ligand", ["none"] * len(cc))
        V["ligand"] = ["+".join(sorted(set([x for x in (a, e) if x != "none"]))) or "none" for a, e in zip(add, emb)]
    if "base" in V and "solvent" in V:                                       # an amine used as the solvent is the base of a condition whose base slot is empty
        V["base"] = [next((a for a in AMINE_SOLVENTS if re.search(r"(^|[^A-Za-z0-9])" + re.escape(a) + r"($|[^A-Za-z0-9])", sol)), "none") if b == "none" else b
                     for b, sol in zip(V["base"], V["solvent"])]
    out = pd.DataFrame(V, index=cond_ids)
    out = out[[c for c in out.columns if out[c].nunique() > 1]]           # a variable with one level is not a variable
    return out

"""
DESCRIPTION:
Given a sequence/regex to find, select those
matching amino acids in the protein.

USAGE:
findseq needle, haystack[, selName[, het[, firstOnly]]]

PARAMS:
needle (string)
		the sequence of amino acids to match and select
		in the haystack.  This can be a sequence of amino
		acids, or a string-style regular expression.  See
		examples.

hastack (string or PyMOL selection)
		name of the PyMOL object/selection in which
		to find the needle.

selName (string; defaults to None)
		This is the name of the selection to return.  If selName
		is left blank (None), then the selection name will be
		foundSeqXYZ where XYZ is some random number; if selName is
		"sele" the usual PyMOL "(sele)" will be used; and, lastly,
		if selName is anything else, that name will be used verbatim.

het (0 or 1; defaults to 0)
		This boolean flag allows (1) or disallows (0) heteroatoms
		from being considered.

firstOnly (0 or 1; defaults to 0)
		Subsequences or motifs might be repeated, this controls how we
		consider multiple matches.  If firstOnly is False (0) then we return
		all found subsequences; if firstOnly is True (1), then we just return
		the first found sequence.

RETURNS:
a newly created selection with the atoms you sought.  If there are
more than two contiguous regions, then a newly created group is
returned with each contiguous segment its own selection.

EXAMPLE:
# find SPVI in 1h12, foundSeqXYZ as return name
findseq SPVI, 1h12

# find FATEW and make it (sele).
findseq FATEW, 1g01, sele

# find the regular expression GMS.*QWY in 1a3h
# and put the return value in (sele).
fetch 1a3h
# this ends up finding the sequence, GMSSHGLQWY
findseq GMS.*QWY, 1a3h, sele

NOTES:
Assumes we're using the ONE LETTER amino acid abbreviations.

AUTHOR:
Jason Vertrees, 2009.
"""

from pymol import cmd
import re
import types
import random

one_letter = {
        '00C': 'C', '01W': 'X', '0A0': 'D', '0A1': 'Y', '0A2': 'K',
        '0A8': 'C', '0AA': 'V', '0AB': 'V', '0AC': 'G', '0AD': 'G',
        '0AF': 'W', '0AG': 'L', '0AH': 'S', '0AK': 'D', '0AM': 'A',
        '0AP': 'C', '0AU': 'U', '0AV': 'A', '0AZ': 'P', '0BN': 'F',
        '0C ': 'C', '0CS': 'A', '0DC': 'C', '0DG': 'G', '0DT': 'T',
        '0G ': 'G', '0NC': 'A', '0SP': 'A', '0U ': 'U', '0YG': 'YG',
        '10C': 'C', '125': 'U', '126': 'U', '127': 'U', '128': 'N',
        '12A': 'A', '143': 'C', '175': 'ASG', '193': 'X', '1AP': 'A',
        '1MA': 'A', '1MG': 'G', '1PA': 'F', '1PI': 'A', '1PR': 'N',
        '1SC': 'C', '1TQ': 'W', '1TY': 'Y', '200': 'F', '23F': 'F',
        '23S': 'X', '26B': 'T', '2AD': 'X', '2AG': 'G', '2AO': 'X',
        '2AR': 'A', '2AS': 'X', '2AT': 'T', '2AU': 'U', '2BD': 'I',
        '2BT': 'T', '2BU': 'A', '2CO': 'C', '2DA': 'A', '2DF': 'N',
        '2DM': 'N', '2DO': 'X', '2DT': 'T', '2EG': 'G', '2FE': 'N',
        '2FI': 'N', '2FM': 'M', '2GT': 'T', '2HF': 'H', '2LU': 'L',
        '2MA': 'A', '2MG': 'G', '2ML': 'L', '2MR': 'R', '2MT': 'P',
        '2MU': 'U', '2NT': 'T', '2OM': 'U', '2OT': 'T', '2PI': 'X',
        '2PR': 'G', '2SA': 'N', '2SI': 'X', '2ST': 'T', '2TL': 'T',
        '2TY': 'Y', '2VA': 'V', '32S': 'X', '32T': 'X', '3AH': 'H',
        '3AR': 'X', '3CF': 'F', '3DA': 'A', '3DR': 'N', '3GA': 'A',
        '3MD': 'D', '3ME': 'U', '3NF': 'Y', '3TY': 'X', '3XH': 'G',
        '4AC': 'N', '4BF': 'Y', '4CF': 'F', '4CY': 'M', '4DP': 'W',
        '4F3': 'GYG', '4FB': 'P', '4FW': 'W', '4HT': 'W', '4IN': 'X',
        '4MF': 'N', '4MM': 'X', '4OC': 'C', '4PC': 'C', '4PD': 'C',
        '4PE': 'C', '4PH': 'F', '4SC': 'C', '4SU': 'U', '4TA': 'N',
        '5AA': 'A', '5AT': 'T', '5BU': 'U', '5CG': 'G', '5CM': 'C',
        '5CS': 'C', '5FA': 'A', '5FC': 'C', '5FU': 'U', '5HP': 'E',
        '5HT': 'T', '5HU': 'U', '5IC': 'C', '5IT': 'T', '5IU': 'U',
        '5MC': 'C', '5MD': 'N', '5MU': 'U', '5NC': 'C', '5PC': 'C',
        '5PY': 'T', '5SE': 'U', '5ZA': 'TWG', '64T': 'T', '6CL': 'K',
        '6CT': 'T', '6CW': 'W', '6HA': 'A', '6HC': 'C', '6HG': 'G',
        '6HN': 'K', '6HT': 'T', '6IA': 'A', '6MA': 'A', '6MC': 'A',
        '6MI': 'N', '6MT': 'A', '6MZ': 'N', '6OG': 'G', '70U': 'U',
        '7DA': 'A', '7GU': 'G', '7JA': 'I', '7MG': 'G', '8AN': 'A',
        '8FG': 'G', '8MG': 'G', '8OG': 'G', '9NE': 'E', '9NF': 'F',
        '9NR': 'R', '9NV': 'V', 'A  ': 'A', 'A1P': 'N', 'A23': 'A',
        'A2L': 'A', 'A2M': 'A', 'A34': 'A', 'A35': 'A', 'A38': 'A',
        'A39': 'A', 'A3A': 'A', 'A3P': 'A', 'A40': 'A', 'A43': 'A',
        'A44': 'A', 'A47': 'A', 'A5L': 'A', 'A5M': 'C', 'A5O': 'A',
        'A66': 'X', 'AA3': 'A', 'AA4': 'A', 'AAR': 'R', 'AB7': 'X',
        'ABA': 'A', 'ABR': 'A', 'ABS': 'A', 'ABT': 'N', 'ACB': 'D',
        'ACL': 'R', 'AD2': 'A', 'ADD': 'X', 'ADX': 'N', 'AEA': 'X',
        'AEI': 'D', 'AET': 'A', 'AFA': 'N', 'AFF': 'N', 'AFG': 'G',
        'AGM': 'R', 'AGT': 'X', 'AHB': 'N', 'AHH': 'X', 'AHO': 'A',
        'AHP': 'A', 'AHS': 'X', 'AHT': 'X', 'AIB': 'A', 'AKL': 'D',
        'ALA': 'A', 'ALC': 'A', 'ALG': 'R', 'ALM': 'A', 'ALN': 'A',
        'ALO': 'T', 'ALQ': 'X', 'ALS': 'A', 'ALT': 'A', 'ALY': 'K',
        'AP7': 'A', 'APE': 'X', 'APH': 'A', 'API': 'K', 'APK': 'K',
        'APM': 'X', 'APP': 'X', 'AR2': 'R', 'AR4': 'E', 'ARG': 'R',
        'ARM': 'R', 'ARO': 'R', 'ARV': 'X', 'AS ': 'A', 'AS2': 'D',
        'AS9': 'X', 'ASA': 'D', 'ASB': 'D', 'ASI': 'D', 'ASK': 'D',
        'ASL': 'D', 'ASM': 'X', 'ASN': 'N', 'ASP': 'D', 'ASQ': 'D',
        'ASU': 'N', 'ASX': 'B', 'ATD': 'T', 'ATL': 'T', 'ATM': 'T',
        'AVC': 'A', 'AVN': 'X', 'AYA': 'A', 'AYG': 'AYG', 'AZK': 'K',
        'AZS': 'S', 'AZY': 'Y', 'B1F': 'F', 'B1P': 'N', 'B2A': 'A',
        'B2F': 'F', 'B2I': 'I', 'B2V': 'V', 'B3A': 'A', 'B3D': 'D',
        'B3E': 'E', 'B3K': 'K', 'B3L': 'X', 'B3M': 'X', 'B3Q': 'X',
        'B3S': 'S', 'B3T': 'X', 'B3U': 'H', 'B3X': 'N', 'B3Y': 'Y',
        'BB6': 'C', 'BB7': 'C', 'BB9': 'C', 'BBC': 'C', 'BCS': 'C',
        'BCX': 'C', 'BE2': 'X', 'BFD': 'D', 'BG1': 'S', 'BGM': 'G',
        'BHD': 'D', 'BIF': 'F', 'BIL': 'X', 'BIU': 'I', 'BJH': 'X',
        'BLE': 'L', 'BLY': 'K', 'BMP': 'N', 'BMT': 'T', 'BNN': 'A',
        'BNO': 'X', 'BOE': 'T', 'BOR': 'R', 'BPE': 'C', 'BRU': 'U',
        'BSE': 'S', 'BT5': 'N', 'BTA': 'L', 'BTC': 'C', 'BTR': 'W',
        'BUC': 'C', 'BUG': 'V', 'BVP': 'U', 'BZG': 'N', 'C  ': 'C',
        'C12': 'TYG', 'C1X': 'K', 'C25': 'C', 'C2L': 'C', 'C2S': 'C',
        'C31': 'C', 'C32': 'C', 'C34': 'C', 'C36': 'C', 'C37': 'C',
        'C38': 'C', 'C3Y': 'C', 'C42': 'C', 'C43': 'C', 'C45': 'C',
        'C46': 'C', 'C49': 'C', 'C4R': 'C', 'C4S': 'C', 'C5C': 'C',
        'C66': 'X', 'C6C': 'C', 'C99': 'TFG', 'CAF': 'C', 'CAL': 'X',
        'CAR': 'C', 'CAS': 'C', 'CAV': 'X', 'CAY': 'C', 'CB2': 'C',
        'CBR': 'C', 'CBV': 'C', 'CCC': 'C', 'CCL': 'K', 'CCS': 'C',
        'CCY': 'CYG', 'CDE': 'X', 'CDV': 'X', 'CDW': 'C', 'CEA': 'C',
        'CFL': 'C', 'CFY': 'FCYG', 'CG1': 'G', 'CGA': 'E', 'CGU': 'E',
        'CH ': 'C', 'CH6': 'MYG', 'CH7': 'KYG', 'CHF': 'X', 'CHG': 'X',
        'CHP': 'G', 'CHS': 'X', 'CIR': 'R', 'CJO': 'GYG', 'CLE': 'L',
        'CLG': 'K', 'CLH': 'K', 'CLV': 'AFG', 'CM0': 'N', 'CME': 'C',
        'CMH': 'C', 'CML': 'C', 'CMR': 'C', 'CMT': 'C', 'CNU': 'U',
        'CP1': 'C', 'CPC': 'X', 'CPI': 'X', 'CQR': 'GYG', 'CR0': 'TLG',
        'CR2': 'GYG', 'CR5': 'G', 'CR7': 'KYG', 'CR8': 'HYG', 'CRF': 'TWG',
        'CRG': 'THG', 'CRK': 'MYG', 'CRO': 'GYG', 'CRQ': 'QYG', 'CRU': 'E',
        'CRW': 'ASG', 'CRX': 'ASG', 'CS0': 'C', 'CS1': 'C', 'CS3': 'C',
        'CS4': 'C', 'CS8': 'N', 'CSA': 'C', 'CSB': 'C', 'CSD': 'C',
        'CSE': 'C', 'CSF': 'C', 'CSH': 'SHG', 'CSI': 'G', 'CSJ': 'C',
        'CSL': 'C', 'CSO': 'C', 'CSP': 'C', 'CSR': 'C', 'CSS': 'C',
        'CSU': 'C', 'CSW': 'C', 'CSX': 'C', 'CSY': 'SYG', 'CSZ': 'C',
        'CTE': 'W', 'CTG': 'T', 'CTH': 'T', 'CUC': 'X', 'CWR': 'S',
        'CXM': 'M', 'CY0': 'C', 'CY1': 'C', 'CY3': 'C', 'CY4': 'C',
        'CYA': 'C', 'CYD': 'C', 'CYF': 'C', 'CYG': 'C', 'CYJ': 'X',
        'CYM': 'C', 'CYQ': 'C', 'CYR': 'C', 'CYS': 'C', 'CZ2': 'C',
        'CZO': 'GYG', 'CZZ': 'C', 'D11': 'T', 'D1P': 'N', 'D3 ': 'N',
        'D33': 'N', 'D3P': 'G', 'D3T': 'T', 'D4M': 'T', 'D4P': 'X',
        'DA ': 'A', 'DA2': 'X', 'DAB': 'A', 'DAH': 'F', 'DAL': 'A',
        'DAR': 'R', 'DAS': 'D', 'DBB': 'T', 'DBM': 'N', 'DBS': 'S',
        'DBU': 'T', 'DBY': 'Y', 'DBZ': 'A', 'DC ': 'C', 'DC2': 'C',
        'DCG': 'G', 'DCI': 'X', 'DCL': 'X', 'DCT': 'C', 'DCY': 'C',
        'DDE': 'H', 'DDG': 'G', 'DDN': 'U', 'DDX': 'N', 'DFC': 'C',
        'DFG': 'G', 'DFI': 'X', 'DFO': 'X', 'DFT': 'N', 'DG ': 'G',
        'DGH': 'G', 'DGI': 'G', 'DGL': 'E', 'DGN': 'Q', 'DHA': 'A',
        'DHI': 'H', 'DHL': 'X', 'DHN': 'V', 'DHP': 'X', 'DHU': 'U',
        'DHV': 'V', 'DI ': 'I', 'DIL': 'I', 'DIR': 'R', 'DIV': 'V',
        'DLE': 'L', 'DLS': 'K', 'DLY': 'K', 'DM0': 'K', 'DMH': 'N',
        'DMK': 'D', 'DMT': 'X', 'DN ': 'N', 'DNE': 'L', 'DNG': 'L',
        'DNL': 'K', 'DNM': 'L', 'DNP': 'A', 'DNR': 'C', 'DNS': 'K',
        'DOA': 'X', 'DOC': 'C', 'DOH': 'D', 'DON': 'L', 'DPB': 'T',
        'DPH': 'F', 'DPL': 'P', 'DPP': 'A', 'DPQ': 'Y', 'DPR': 'P',
        'DPY': 'N', 'DRM': 'U', 'DRP': 'N', 'DRT': 'T', 'DRZ': 'N',
        'DSE': 'S', 'DSG': 'N', 'DSN': 'S', 'DSP': 'D', 'DT ': 'T',
        'DTH': 'T', 'DTR': 'W', 'DTY': 'Y', 'DU ': 'U', 'DVA': 'V',
        'DXD': 'N', 'DXN': 'N', 'DYG': 'DYG', 'DYS': 'C', 'DZM': 'A',
        'E  ': 'A', 'E1X': 'A', 'EDA': 'A', 'EDC': 'G', 'EFC': 'C',
        'EHP': 'F', 'EIT': 'T', 'ENP': 'N', 'ESB': 'Y', 'ESC': 'M',
        'EXY': 'L', 'EY5': 'N', 'EYS': 'X', 'F2F': 'F', 'FA2': 'A',
        'FA5': 'N', 'FAG': 'N', 'FAI': 'N', 'FCL': 'F', 'FFD': 'N',
        'FGL': 'G', 'FGP': 'S', 'FHL': 'X', 'FHO': 'K', 'FHU': 'U',
        'FLA': 'A', 'FLE': 'L', 'FLT': 'Y', 'FME': 'M', 'FMG': 'G',
        'FMU': 'N', 'FOE': 'C', 'FOX': 'G', 'FP9': 'P', 'FPA': 'F',
        'FRD': 'X', 'FT6': 'W', 'FTR': 'W', 'FTY': 'Y', 'FZN': 'K',
        'G  ': 'G', 'G25': 'G', 'G2L': 'G', 'G2S': 'G', 'G31': 'G',
        'G32': 'G', 'G33': 'G', 'G36': 'G', 'G38': 'G', 'G42': 'G',
        'G46': 'G', 'G47': 'G', 'G48': 'G', 'G49': 'G', 'G4P': 'N',
        'G7M': 'G', 'GAO': 'G', 'GAU': 'E', 'GCK': 'C', 'GCM': 'X',
        'GDP': 'G', 'GDR': 'G', 'GFL': 'G', 'GGL': 'E', 'GH3': 'G',
        'GHG': 'Q', 'GHP': 'G', 'GL3': 'G', 'GLH': 'Q', 'GLM': 'X',
        'GLN': 'Q', 'GLQ': 'E', 'GLU': 'E', 'GLX': 'Z', 'GLY': 'G',
        'GLZ': 'G', 'GMA': 'E', 'GMS': 'G', 'GMU': 'U', 'GN7': 'G',
        'GND': 'X', 'GNE': 'N', 'GOM': 'G', 'GPL': 'K', 'GS ': 'G',
        'GSC': 'G', 'GSR': 'G', 'GSS': 'G', 'GSU': 'E', 'GT9': 'C',
        'GTP': 'G', 'GVL': 'X', 'GYC': 'CYG', 'GYS': 'SYG', 'H2U': 'U',
        'H5M': 'P', 'HAC': 'A', 'HAR': 'R', 'HBN': 'H', 'HCS': 'X',
        'HDP': 'U', 'HEU': 'U', 'HFA': 'X', 'HGL': 'X', 'HHI': 'H',
        'HHK': 'AK', 'HIA': 'H', 'HIC': 'H', 'HIP': 'H', 'HIQ': 'H',
        'HIS': 'H', 'HL2': 'L', 'HLU': 'L', 'HMF': 'A', 'HMR': 'R',
        'HOL': 'N', 'HPC': 'F', 'HPE': 'F', 'HPQ': 'F', 'HQA': 'A',
        'HRG': 'R', 'HRP': 'W', 'HS8': 'H', 'HS9': 'H', 'HSE': 'S',
        'HSL': 'S', 'HSO': 'H', 'HTI': 'C', 'HTN': 'N', 'HTR': 'W',
        'HV5': 'A', 'HVA': 'V', 'HY3': 'P', 'HYP': 'P', 'HZP': 'P',
        'I  ': 'I', 'I2M': 'I', 'I58': 'K', 'I5C': 'C', 'IAM': 'A',
        'IAR': 'R', 'IAS': 'D', 'IC ': 'C', 'IEL': 'K', 'IEY': 'HYG',
        'IG ': 'G', 'IGL': 'G', 'IGU': 'G', 'IIC': 'SHG', 'IIL': 'I',
        'ILE': 'I', 'ILG': 'E', 'ILX': 'I', 'IMC': 'C', 'IML': 'I',
        'IOY': 'F', 'IPG': 'G', 'IPN': 'N', 'IRN': 'N', 'IT1': 'K',
        'IU ': 'U', 'IYR': 'Y', 'IYT': 'T', 'JJJ': 'C', 'JJK': 'C',
        'JJL': 'C', 'JW5': 'N', 'K1R': 'C', 'KAG': 'G', 'KCX': 'K',
        'KGC': 'K', 'KOR': 'M', 'KPI': 'K', 'KST': 'K', 'KYQ': 'K',
        'L2A': 'X', 'LA2': 'K', 'LAA': 'D', 'LAL': 'A', 'LBY': 'K',
        'LC ': 'C', 'LCA': 'A', 'LCC': 'N', 'LCG': 'G', 'LCH': 'N',
        'LCK': 'K', 'LCX': 'K', 'LDH': 'K', 'LED': 'L', 'LEF': 'L',
        'LEH': 'L', 'LEI': 'V', 'LEM': 'L', 'LEN': 'L', 'LET': 'X',
        'LEU': 'L', 'LG ': 'G', 'LGP': 'G', 'LHC': 'X', 'LHU': 'U',
        'LKC': 'N', 'LLP': 'K', 'LLY': 'K', 'LME': 'E', 'LMQ': 'Q',
        'LMS': 'N', 'LP6': 'K', 'LPD': 'P', 'LPG': 'G', 'LPL': 'X',
        'LPS': 'S', 'LSO': 'X', 'LTA': 'X', 'LTR': 'W', 'LVG': 'G',
        'LVN': 'V', 'LYM': 'K', 'LYN': 'K', 'LYR': 'K', 'LYS': 'K',
        'LYX': 'K', 'LYZ': 'K', 'M0H': 'C', 'M1G': 'G', 'M2G': 'G',
        'M2L': 'K', 'M2S': 'M', 'M3L': 'K', 'M5M': 'C', 'MA ': 'A',
        'MA6': 'A', 'MA7': 'A', 'MAA': 'A', 'MAD': 'A', 'MAI': 'R',
        'MBQ': 'Y', 'MBZ': 'N', 'MC1': 'S', 'MCG': 'X', 'MCL': 'K',
        'MCS': 'C', 'MCY': 'C', 'MDH': 'X', 'MDO': 'ASG', 'MDR': 'N',
        'MEA': 'F', 'MED': 'M', 'MEG': 'E', 'MEN': 'N', 'MEP': 'U',
        'MEQ': 'Q', 'MET': 'M', 'MEU': 'G', 'MF3': 'X', 'MFC': 'GYG',
        'MG1': 'G', 'MGG': 'R', 'MGN': 'Q', 'MGQ': 'A', 'MGV': 'G',
        'MGY': 'G', 'MHL': 'L', 'MHO': 'M', 'MHS': 'H', 'MIA': 'A',
        'MIS': 'S', 'MK8': 'L', 'ML3': 'K', 'MLE': 'L', 'MLL': 'L',
        'MLY': 'K', 'MLZ': 'K', 'MME': 'M', 'MMT': 'T', 'MND': 'N',
        'MNL': 'L', 'MNU': 'U', 'MNV': 'V', 'MOD': 'X', 'MP8': 'P',
        'MPH': 'X', 'MPJ': 'X', 'MPQ': 'G', 'MRG': 'G', 'MSA': 'G',
        'MSE': 'M', 'MSL': 'M', 'MSO': 'M', 'MSP': 'X', 'MT2': 'M',
        'MTR': 'T', 'MTU': 'A', 'MTY': 'Y', 'MVA': 'V', 'N  ': 'N',
        'N10': 'S', 'N2C': 'X', 'N5I': 'N', 'N5M': 'C', 'N6G': 'G',
        'N7P': 'P', 'NA8': 'A', 'NAL': 'A', 'NAM': 'A', 'NB8': 'N',
        'NBQ': 'Y', 'NC1': 'S', 'NCB': 'A', 'NCX': 'N', 'NCY': 'X',
        'NDF': 'F', 'NDN': 'U', 'NEM': 'H', 'NEP': 'H', 'NF2': 'N',
        'NFA': 'F', 'NHL': 'E', 'NIT': 'X', 'NIY': 'Y', 'NLE': 'L',
        'NLN': 'L', 'NLO': 'L', 'NLP': 'L', 'NLQ': 'Q', 'NMC': 'G',
        'NMM': 'R', 'NMS': 'T', 'NMT': 'T', 'NNH': 'R', 'NP3': 'N',
        'NPH': 'C', 'NRP': 'LYG', 'NRQ': 'MYG', 'NSK': 'X', 'NTY': 'Y',
        'NVA': 'V', 'NYC': 'TWG', 'NYG': 'NYG', 'NYM': 'N', 'NYS': 'C',
        'NZH': 'H', 'O12': 'X', 'O2C': 'N', 'O2G': 'G', 'OAD': 'N',
        'OAS': 'S', 'OBF': 'X', 'OBS': 'X', 'OCS': 'C', 'OCY': 'C',
        'ODP': 'N', 'OHI': 'H', 'OHS': 'D', 'OIC': 'X', 'OIP': 'I',
        'OLE': 'X', 'OLT': 'T', 'OLZ': 'S', 'OMC': 'C', 'OMG': 'G',
        'OMT': 'M', 'OMU': 'U', 'ONE': 'U', 'ONL': 'X', 'OPR': 'R',
        'ORN': 'A', 'ORQ': 'R', 'OSE': 'S', 'OTB': 'X', 'OTH': 'T',
        'OTY': 'Y', 'OXX': 'D', 'P  ': 'G', 'P1L': 'C', 'P1P': 'N',
        'P2T': 'T', 'P2U': 'U', 'P2Y': 'P', 'P5P': 'A', 'PAQ': 'Y',
        'PAS': 'D', 'PAT': 'W', 'PAU': 'A', 'PBB': 'C', 'PBF': 'F',
        'PBT': 'N', 'PCA': 'E', 'PCC': 'P', 'PCE': 'X', 'PCS': 'F',
        'PDL': 'X', 'PDU': 'U', 'PEC': 'C', 'PF5': 'F', 'PFF': 'F',
        'PFX': 'X', 'PG1': 'S', 'PG7': 'G', 'PG9': 'G', 'PGL': 'X',
        'PGN': 'G', 'PGP': 'G', 'PGY': 'G', 'PHA': 'F', 'PHD': 'D',
        'PHE': 'F', 'PHI': 'F', 'PHL': 'F', 'PHM': 'F', 'PIV': 'X',
        'PLE': 'L', 'PM3': 'F', 'PMT': 'C', 'POM': 'P', 'PPN': 'F',
        'PPU': 'A', 'PPW': 'G', 'PQ1': 'N', 'PR3': 'C', 'PR5': 'A',
        'PR9': 'P', 'PRN': 'A', 'PRO': 'P', 'PRS': 'P', 'PSA': 'F',
        'PSH': 'H', 'PST': 'T', 'PSU': 'U', 'PSW': 'C', 'PTA': 'X',
        'PTH': 'Y', 'PTM': 'Y', 'PTR': 'Y', 'PU ': 'A', 'PUY': 'N',
        'PVH': 'H', 'PVL': 'X', 'PYA': 'A', 'PYO': 'U', 'PYX': 'C',
        'PYY': 'N', 'QLG': 'QLG', 'QUO': 'G', 'R  ': 'A', 'R1A': 'C',
        'R1B': 'C', 'R1F': 'C', 'R7A': 'C', 'RC7': 'HYG', 'RCY': 'C',
        'RIA': 'A', 'RMP': 'A', 'RON': 'X', 'RT ': 'T', 'RTP': 'N',
        'S1H': 'S', 'S2C': 'C', 'S2D': 'A', 'S2M': 'T', 'S2P': 'A',
        'S4A': 'A', 'S4C': 'C', 'S4G': 'G', 'S4U': 'U', 'S6G': 'G',
        'SAC': 'S', 'SAH': 'C', 'SAR': 'G', 'SBL': 'S', 'SC ': 'C',
        'SCH': 'C', 'SCS': 'C', 'SCY': 'C', 'SD2': 'X', 'SDG': 'G',
        'SDP': 'S', 'SEB': 'S', 'SEC': 'A', 'SEG': 'A', 'SEL': 'S',
        'SEM': 'X', 'SEN': 'S', 'SEP': 'S', 'SER': 'S', 'SET': 'S',
        'SGB': 'S', 'SHC': 'C', 'SHP': 'G', 'SHR': 'K', 'SIB': 'C',
        'SIC': 'DC', 'SLA': 'P', 'SLR': 'P', 'SLZ': 'K', 'SMC': 'C',
        'SME': 'M', 'SMF': 'F', 'SMP': 'A', 'SMT': 'T', 'SNC': 'C',
        'SNN': 'N', 'SOC': 'C', 'SOS': 'N', 'SOY': 'S', 'SPT': 'T',
        'SRA': 'A', 'SSU': 'U', 'STY': 'Y', 'SUB': 'X', 'SUI': 'DG',
        'SUN': 'S', 'SUR': 'U', 'SVA': 'S', 'SVX': 'S', 'SVZ': 'X',
        'SYS': 'C', 'T  ': 'T', 'T11': 'F', 'T23': 'T', 'T2S': 'T',
        'T2T': 'N', 'T31': 'U', 'T32': 'T', 'T36': 'T', 'T37': 'T',
        'T38': 'T', 'T39': 'T', 'T3P': 'T', 'T41': 'T', 'T48': 'T',
        'T49': 'T', 'T4S': 'T', 'T5O': 'U', 'T5S': 'T', 'T66': 'X',
        'T6A': 'A', 'TA3': 'T', 'TA4': 'X', 'TAF': 'T', 'TAL': 'N',
        'TAV': 'D', 'TBG': 'V', 'TBM': 'T', 'TC1': 'C', 'TCP': 'T',
        'TCQ': 'X', 'TCR': 'W', 'TCY': 'A', 'TDD': 'L', 'TDY': 'T',
        'TFE': 'T', 'TFO': 'A', 'TFQ': 'F', 'TFT': 'T', 'TGP': 'G',
        'TH6': 'T', 'THC': 'T', 'THO': 'X', 'THR': 'T', 'THX': 'N',
        'THZ': 'R', 'TIH': 'A', 'TLB': 'N', 'TLC': 'T', 'TLN': 'U',
        'TMB': 'T', 'TMD': 'T', 'TNB': 'C', 'TNR': 'S', 'TOX': 'W',
        'TP1': 'T', 'TPC': 'C', 'TPG': 'G', 'TPH': 'X', 'TPL': 'W',
        'TPO': 'T', 'TPQ': 'Y', 'TQQ': 'W', 'TRF': 'W', 'TRG': 'K',
        'TRN': 'W', 'TRO': 'W', 'TRP': 'W', 'TRQ': 'W', 'TRW': 'W',
        'TRX': 'W', 'TS ': 'N', 'TST': 'X', 'TT ': 'N', 'TTD': 'T',
        'TTI': 'U', 'TTM': 'T', 'TTQ': 'W', 'TTS': 'Y', 'TY2': 'Y',
        'TY3': 'Y', 'TYB': 'Y', 'TYI': 'Y', 'TYN': 'Y', 'TYO': 'Y',
        'TYQ': 'Y', 'TYR': 'Y', 'TYS': 'Y', 'TYT': 'Y', 'TYU': 'N',
        'TYX': 'X', 'TYY': 'Y', 'TZB': 'X', 'TZO': 'X', 'U  ': 'U',
        'U25': 'U', 'U2L': 'U', 'U2N': 'U', 'U2P': 'U', 'U31': 'U',
        'U33': 'U', 'U34': 'U', 'U36': 'U', 'U37': 'U', 'U8U': 'U',
        'UAR': 'U', 'UCL': 'U', 'UD5': 'U', 'UDP': 'N', 'UFP': 'N',
        'UFR': 'U', 'UFT': 'U', 'UMA': 'A', 'UMP': 'U', 'UMS': 'U',
        'UN1': 'X', 'UN2': 'X', 'UNK': 'X', 'UR3': 'U', 'URD': 'U',
        'US1': 'U', 'US2': 'U', 'US3': 'T', 'US5': 'U', 'USM': 'U',
        'V1A': 'C', 'VAD': 'V', 'VAF': 'V', 'VAL': 'V', 'VB1': 'K',
        'VDL': 'X', 'VLL': 'X', 'VLM': 'X', 'VMS': 'X', 'VOL': 'X',
        'X  ': 'G', 'X2W': 'E', 'X4A': 'N', 'X9Q': 'AFG', 'XAD': 'A',
        'XAE': 'N', 'XAL': 'A', 'XAR': 'N', 'XCL': 'C', 'XCP': 'X',
        'XCR': 'C', 'XCS': 'N', 'XCT': 'C', 'XCY': 'C', 'XGA': 'N',
        'XGL': 'G', 'XGR': 'G', 'XGU': 'G', 'XTH': 'T', 'XTL': 'T',
        'XTR': 'T', 'XTS': 'G', 'XTY': 'N', 'XUA': 'A', 'XUG': 'G',
        'XX1': 'K', 'XXY': 'THG', 'XYG': 'DYG', 'Y  ': 'A', 'YCM': 'C',
        'YG ': 'G', 'YOF': 'Y', 'YRR': 'N', 'YYG': 'G', 'Z  ': 'C',
        'ZAD': 'A', 'ZAL': 'A', 'ZBC': 'C', 'ZCY': 'C', 'ZDU': 'U',
        'ZFB': 'X', 'ZGU': 'G', 'ZHP': 'N', 'ZTH': 'T', 'ZZJ': 'A'}
nucleotides_one_letter = {
        'A':'Adenine','C':'Cytosine','G':'Guanine','T':'Thymine','U':'Uracil','M':'A or C(amino)','R':'A or G(purine)',
        'W':'A or T(weak)','S':'C or G(strong)','Y':'C or T(pyrimidine)','K':'G or T(keto)','V':'A or C or G','H':'A or C or T',
        'D':'A or G or T','B':'C or G or T','N':'A or G or C or T(any)'
    }
from chempy import cpv
def get_coord(v,cmd):
    if not isinstance(v, str):
        try:
            return v[:3]
        except:
            return False
    if v.startswith('['):
        return cmd.safe_list_eval(v)[:3]
    try:
        if cmd.count_atoms(v)==1:
            # atom coordinates
            return cmd.get_atom_coords(v)
        else:
            # more than one atom --> use "center"
            # alt check!
            if cmd.count_atoms('(alt *) and not (alt "")')!=0:
                print("distancetoatom: warning! alternative coordinates found for origin, using center!")
            view_temp=cmd.get_view()
            cmd.zoom(v)
            v=cmd.get_position()
            cmd.set_view(view_temp)
            return v
    except:
        return False

import sys
from pymol import stored

def distancetoatom(origin='pk1',cutoff=10,filename=None,selection='all',state=0,property_name='p.dist',coordinates=0,decimals=3,sort=1,quiet=1,cmd=None,origin_atom_obj=None,model_name=None,hla_name=None,ref_sequence=None):
    '''
DESCRIPTION

    distancetoatom.py
    Described at: http://www.pymolwiki.org/Distancetoatom

    Prints all distanced between the specified atom/coordinate/center
    and all atoms within cutoff distance that are part of the selection.
    All coordinates and distances can be saved in a csv-style text file report
    and can be appended to a (custom) atom property, if defined.

USAGE

    distancetoatom [ origin [, cutoff [, filename [, selection
    [, state [, property_name [, coordinates [, decimals [, sort
    [, quiet ]]]]]]]]]]

ARGUMENTS

    NAME        TYPE    FUNCTION
    origin:     <list>  defines the coordinates for the origin and can be:
                <str>   1. a list with coordinates [x,y,z]
                        2. a single atom selection string {default='pk1'}
                        3. a multi-atom selection string (center will be used)
    cutoff      <float> sets the maximum distance {default: 10}
    filename    <str>   filename for optional output report. {default=None}
                        set to e.g. 'report.txt' to create a report
                        (omit or set to '', None, 0 or False to disable)
    selection   <str>   can be used to define/limit the measurment to specific
                        sub-selections {default='all'}
    state       <int>   object state, {default=0} # = current
    property_name <str> the distance will be stored in this property {p.dist}
                        set "" to disable
    coordinates <int>   toggle whether atom coordinated will be reported {0}
    decimals    <int>   decimals for coordinates and distance:
                        default = 3 # = max. PDB resolution
    sort        <int>   Sorting by distance?
                         1: ascending (default)
                         0: no sorting (by names)
                        -1: descending
    quiet       <bool>  toggle verbosity
    '''
    # keyword check
    try:
        selection = '(%s)'%selection
        ori=get_coord(origin,cmd)
        if not ori:
            print("distancetoatom: aborting - check input for 'origin'!")
            return False
        cutoff = abs(float(cutoff))
        filename = str(filename)
        state = abs(int(state))
        if (not int(state)):
            state=cmd.get_state()
        # cmd.set('state', state) # distance by state
        property_name = str(property_name)
        decimals = abs(int(decimals))
        sort = int(sort)
        coordinates=bool(int(coordinates))
        quiet=bool(int(quiet))
    except:
        print('distancetoatom: aborting - input error!')
        return False

    # round origin
    ori = [round(x,decimals) for x in origin]

    # # report?
    # if filename in ['', '0', 'False', 'None']:
    #     filename=False
    # else:
    #     try:
    #         report=open(filename,'w') # file for writing
    #     except:
    #         print('distancetoatom: Unable to open report file! - Aborting!')
    #         return False

    # temporary name for pseudoatom
    tempname = cmd.get_unused_name('temp_name')
    tempsel = cmd.get_unused_name('temp_sel')

    #origin
    cmd.pseudoatom(object=tempname, resi=1, pos=ori)

    # select atoms within cutoff
    cmd.select(tempsel, '(%s around %f) and (%s) and state %d' %(tempname, cutoff, selection, state))
    cmd.delete(tempname)

    # # single atom ori and part of selection
    # # avoid double reporting!
    # single_atom_ori=False
    # try:
    #     if cmd.count_atoms('(%s) and (%s) and (%s)'%(selection, tempsel, origin))==1:
    #         single_atom_ori=True
    # except: pass
    # pass= coordinates or multi-atom or single, not selected --> report ori

    # atom list
    stored.temp=[]
    cmd.iterate(tempsel, 'stored.temp.append([model, segi, chain, resn, resi, name, alt,index])')

    # custom properties? # conditional disabling
    if (property_name==''): property_name=False
    if ((cmd.get_version()[1]<1.7) and (property_name not in ['b','q'])):
        property_name=False

    # calculate the distances, creating list
    distance_list=[]
    # if (not single_atom_ori):
    #     distance_list.append(['ORIGIN: '+str(origin), ori[0], ori[1], ori[2], 0.0])

    for atom in stored.temp:
        atom_name = ('/%s/%s/%s/%s`%s/%s`%s'%(atom[0], atom[1], atom[2], atom[3], atom[4], atom[5], atom[6]))
        # atom_xyz = [round(x, decimals) for x in cmd.get_atom_coords(atom_name)]
        # atom_dist = round(cpv.distance(ori, atom_xyz), decimals)
        if origin_atom_obj and model_name:

            origin_atom_name=('/%s/%s/%s/%s`%s/%s`%s'%(model_name,origin_atom_obj.segi, origin_atom_obj.chain, origin_atom_obj.resn, origin_atom_obj.resi,origin_atom_obj.name, origin_atom_obj.alt))
            # print('calc dist')
            d = round(cmd.distance("temp_distance",origin_atom_name,atom_name,mode=1),decimals)
            if d>0:
                # target_atom_model = cmd.get_model(atom_name)
                # target_atom_index = [a.index for a in target_atom_model.atom][0]
                target_node = 'amino_{}_{}_hla_{}_peptide_{}_atom_{}'.format(str(atom[4]), one_letter[atom[3]],
                                                                             hla_name,
                                                                             ref_sequence,
                                                                             str(atom[-1]))
                distance_list.append([atom_name,  d,target_node])

        # else:
        #     distance_list.append([atom_name,atom_xyz[0],atom_xyz[1],atom_xyz[2], atom_dist])
        # create property with distance (can be used for coloring, labeling etc)
        # if property_name:
        #     try:
        #         cmd.alter(atom_name, '%s=%f'%(property_name, atom_dist))
        #     except:
        #         # I'm not sure alter raises exceptions if the property is illegal
        #         property_name=False

    # # sort list, if selected
    # if sort>0: distance_list.sort(key=lambda dist: dist[2])
    # elif sort<0: distance_list.sort(key=lambda dist: dist[4], reverse=True)
    # # add header
    # distance_list=distance_list

    # if ((not quiet) and (filename)):
    #     # Hijack stdout to print to file and console
    #     class logboth(object):
    #         def __init__(self, *files):
    #             self.files = files
    #         def write(self, obj):
    #             for f in self.files:
    #                 f.write(obj)
    #     originalstdout = sys.stdout
    #     sys.stdout = logboth(sys.stdout, report)
    #
    # for entry in distance_list:
    #     if coordinates:
    #         output= '%s, %s, %s, %s, %s' %(entry[0],entry[1],entry[2],entry[3],entry[4]) #csv style
    #     else:
    #         output= '%s, %s' %(entry[0],entry[4]) #csv style
    #     if (not quiet):
    #         print(output)
    #     elif filename:
    #         report.write(output+'\n')

    # # restore regular stdout
    # if ((not quiet) and (filename)): sys.stdout = originalstdout
    # # close file
    # if filename: report.close()

    # if (not quiet):
    #     if property_name: print('Distances saved to property: %s' %str(property_name))
    #     else: print('Distances NOT saved to property (illegal custom property)')

    # remove temp. selection
    cmd.delete(tempsel)
    cmd.delete('temp_distance')
    # return list for potential use:
    # if coordinates:
    #     if len(distance_list)>2: # prevents crash if list is otherwise empty
    #         distance_list2=list(map(distance_list.__getitem__, [1,4]))
    #         return distance_list2
    #     else: return distance_list
    # else:
    return distance_list
def show_contacts(selection, selection2, result="contacts", cutoff=3.6, bigcutoff=4.0, SC_DEBUG=4,DEBUG=4,cmd=None):
    """
    USAGE

    show_contacts selection, selection2, [result=contacts],[cutoff=3.6],[bigcutoff=4.0]

    Show various polar contacts, the good, the bad, and the ugly.

    Edit MPB 6-26-14: The distances are heavy atom distances, so I upped the default cutoff to 4.0

    Returns:
    True/False -  if False, something went wrong
    """
    if SC_DEBUG > 4:
        print('Starting show_contacts')
        print('selection = "' + selection + '"')
        print('selection2 = "' + selection2 + '"')

    result = cmd.get_legal_name(result)

    # if the group of contacts already exist, delete them
    cmd.delete(result)

    # ensure only N and O atoms are in the selection
    all_don_acc1 = selection + " and (donor or acceptor)"
    all_don_acc2 = selection2 + " and  (donor or acceptor)"

    if SC_DEBUG > 4:
        print('all_don_acc1 = "' + all_don_acc1 + '"')
        print('all_don_acc2 = "' + all_don_acc2 + '"')

    # if theses selections turn out not to have any atoms in them, pymol throws cryptic errors when calling the dist function like:
    # 'Selector-Error: Invalid selection name'
    # So for each one, manually perform the selection and then pass the reference to the distance command and at the end, clean up the selections
    # the return values are the count of the number of atoms
    all1_sele_count = cmd.select('all_don_acc1_sele', all_don_acc1)
    all2_sele_count = cmd.select('all_don_acc2_sele', all_don_acc2)

    # print out some warnings
    if DEBUG > 3:
        if not all1_sele_count:
            print('Warning: all_don_acc1 selection empty!')
        if not all2_sele_count:
            print('Warning: all_don_acc2 selection empty!')

    ########################################
    allres = result + "_all"
    if all1_sele_count and all2_sele_count:
        cmd.distance(allres, 'all_don_acc1_sele', 'all_don_acc2_sele', bigcutoff, mode=0)
        cmd.set("dash_radius", "0.05", allres)
        cmd.set("dash_color", "purple", allres)
        cmd.hide("labels", allres)

    ########################################
    # compute good polar interactions according to pymol
    polres = result + "_polar"
    if all1_sele_count and all2_sele_count:
        cmd.distance(polres, 'all_don_acc1_sele', 'all_don_acc2_sele', cutoff,
                     mode=2)  # hopefully this checks angles? Yes
        cmd.set("dash_radius", "0.126", polres)

    ########################################
    # When running distance in mode=2, the cutoff parameter is ignored if set higher then the default of 3.6
    # so set it to the passed in cutoff and change it back when you are done.
    old_h_bond_cutoff_center = cmd.get('h_bond_cutoff_center')  # ideal geometry
    old_h_bond_cutoff_edge = cmd.get('h_bond_cutoff_edge')  # minimally acceptable geometry
    cmd.set('h_bond_cutoff_center', bigcutoff)
    cmd.set('h_bond_cutoff_edge', bigcutoff)

    # compute possibly suboptimal polar interactions using the user specified distance
    pol_ok_res = result + "_polar_ok"
    if all1_sele_count and all2_sele_count:
        cmd.distance(pol_ok_res, 'all_don_acc1_sele', 'all_don_acc2_sele', bigcutoff, mode=2)
        cmd.set("dash_radius", "0.06", pol_ok_res)

    # now reset the h_bond cutoffs
    cmd.set('h_bond_cutoff_center', old_h_bond_cutoff_center)
    cmd.set('h_bond_cutoff_edge', old_h_bond_cutoff_edge)

    ########################################

    onlyacceptors1 = selection + " and (acceptor and !donor)"
    onlyacceptors2 = selection2 + " and (acceptor and !donor)"
    onlydonors1 = selection + " and (!acceptor and donor)"
    onlydonors2 = selection2 + " and (!acceptor and donor)"

    # perform the selections
    onlyacceptors1_sele_count = cmd.select('onlyacceptors1_sele', onlyacceptors1)
    onlyacceptors2_sele_count = cmd.select('onlyacceptors2_sele', onlyacceptors2)
    onlydonors1_sele_count = cmd.select('onlydonors1_sele', onlydonors1)
    onlydonors2_sele_count = cmd.select('onlydonors2_sele', onlydonors2)

    # print out some warnings
    if SC_DEBUG > 2:
        if not onlyacceptors1_sele_count:
            print('Warning: onlyacceptors1 selection empty!')
        if not onlyacceptors2_sele_count:
            print('Warning: onlyacceptors2 selection empty!')
        if not onlydonors1_sele_count:
            print('Warning: onlydonors1 selection empty!')
        if not onlydonors2_sele_count:
            print('Warning: onlydonors2 selection empty!')

    accres = result + "_aa"
    if onlyacceptors1_sele_count and onlyacceptors2_sele_count:
        aa_dist_out = cmd.distance(accres, 'onlyacceptors1_sele', 'onlyacceptors2_sele', cutoff, 0)

        if aa_dist_out < 0:
            print('\n\nCaught a pymol selection error in acceptor-acceptor selection of show_contacts')
            print('accres:', accres)
            print('onlyacceptors1', onlyacceptors1)
            print('onlyacceptors2', onlyacceptors2)
            return False

        cmd.set("dash_color", "red", accres)
        cmd.set("dash_radius", "0.125", accres)

    ########################################

    donres = result + "_dd"
    if onlydonors1_sele_count and onlydonors2_sele_count:
        dd_dist_out = cmd.distance(donres, 'onlydonors1_sele', 'onlydonors2_sele', cutoff, 0)

        # try to catch the error state
        if dd_dist_out < 0:
            print('\n\nCaught a pymol selection error in dd selection of show_contacts')
            print('donres:', donres)
            print('onlydonors1', onlydonors1)
            print('onlydonors2', onlydonors2)
            print(
                "cmd.distance('" + donres + "', '" + onlydonors1 + "', '" + onlydonors2 + "', " + str(cutoff) + ", 0)")
            return False

        cmd.set("dash_color", "red", donres)
        cmd.set("dash_radius", "0.125", donres)

    ##########################################################
    ##### find the buried unpaired atoms of the receptor #####
    ##########################################################

    # initialize the variable for when CALC_SASA is False
    unpaired_atoms = ''

    ## Group
    cmd.group(result, "%s %s %s %s %s %s" % (polres, allres, accres, donres, pol_ok_res, unpaired_atoms))

    ## Clean up the selection objects
    # if the show_contacts debug level is high enough, don't delete them.
    if SC_DEBUG < 5:
        cmd.delete('all_don_acc1_sele')
        cmd.delete('all_don_acc2_sele')
        cmd.delete('onlyacceptors1_sele')
        cmd.delete('onlyacceptors2_sele')
        cmd.delete('onlydonors1_sele')
        cmd.delete('onlydonors2_sele')

    return True
def find_amino_in_selection_with_cmd(selName=None, cmd=None,het=0, firstOnly=0):

    one_letter = {
        '00C': 'C', '01W': 'X', '0A0': 'D', '0A1': 'Y', '0A2': 'K',
        '0A8': 'C', '0AA': 'V', '0AB': 'V', '0AC': 'G', '0AD': 'G',
        '0AF': 'W', '0AG': 'L', '0AH': 'S', '0AK': 'D', '0AM': 'A',
        '0AP': 'C', '0AU': 'U', '0AV': 'A', '0AZ': 'P', '0BN': 'F',
        '0C ': 'C', '0CS': 'A', '0DC': 'C', '0DG': 'G', '0DT': 'T',
        '0G ': 'G', '0NC': 'A', '0SP': 'A', '0U ': 'U', '0YG': 'YG',
        '10C': 'C', '125': 'U', '126': 'U', '127': 'U', '128': 'N',
        '12A': 'A', '143': 'C', '175': 'ASG', '193': 'X', '1AP': 'A',
        '1MA': 'A', '1MG': 'G', '1PA': 'F', '1PI': 'A', '1PR': 'N',
        '1SC': 'C', '1TQ': 'W', '1TY': 'Y', '200': 'F', '23F': 'F',
        '23S': 'X', '26B': 'T', '2AD': 'X', '2AG': 'G', '2AO': 'X',
        '2AR': 'A', '2AS': 'X', '2AT': 'T', '2AU': 'U', '2BD': 'I',
        '2BT': 'T', '2BU': 'A', '2CO': 'C', '2DA': 'A', '2DF': 'N',
        '2DM': 'N', '2DO': 'X', '2DT': 'T', '2EG': 'G', '2FE': 'N',
        '2FI': 'N', '2FM': 'M', '2GT': 'T', '2HF': 'H', '2LU': 'L',
        '2MA': 'A', '2MG': 'G', '2ML': 'L', '2MR': 'R', '2MT': 'P',
        '2MU': 'U', '2NT': 'T', '2OM': 'U', '2OT': 'T', '2PI': 'X',
        '2PR': 'G', '2SA': 'N', '2SI': 'X', '2ST': 'T', '2TL': 'T',
        '2TY': 'Y', '2VA': 'V', '32S': 'X', '32T': 'X', '3AH': 'H',
        '3AR': 'X', '3CF': 'F', '3DA': 'A', '3DR': 'N', '3GA': 'A',
        '3MD': 'D', '3ME': 'U', '3NF': 'Y', '3TY': 'X', '3XH': 'G',
        '4AC': 'N', '4BF': 'Y', '4CF': 'F', '4CY': 'M', '4DP': 'W',
        '4F3': 'GYG', '4FB': 'P', '4FW': 'W', '4HT': 'W', '4IN': 'X',
        '4MF': 'N', '4MM': 'X', '4OC': 'C', '4PC': 'C', '4PD': 'C',
        '4PE': 'C', '4PH': 'F', '4SC': 'C', '4SU': 'U', '4TA': 'N',
        '5AA': 'A', '5AT': 'T', '5BU': 'U', '5CG': 'G', '5CM': 'C',
        '5CS': 'C', '5FA': 'A', '5FC': 'C', '5FU': 'U', '5HP': 'E',
        '5HT': 'T', '5HU': 'U', '5IC': 'C', '5IT': 'T', '5IU': 'U',
        '5MC': 'C', '5MD': 'N', '5MU': 'U', '5NC': 'C', '5PC': 'C',
        '5PY': 'T', '5SE': 'U', '5ZA': 'TWG', '64T': 'T', '6CL': 'K',
        '6CT': 'T', '6CW': 'W', '6HA': 'A', '6HC': 'C', '6HG': 'G',
        '6HN': 'K', '6HT': 'T', '6IA': 'A', '6MA': 'A', '6MC': 'A',
        '6MI': 'N', '6MT': 'A', '6MZ': 'N', '6OG': 'G', '70U': 'U',
        '7DA': 'A', '7GU': 'G', '7JA': 'I', '7MG': 'G', '8AN': 'A',
        '8FG': 'G', '8MG': 'G', '8OG': 'G', '9NE': 'E', '9NF': 'F',
        '9NR': 'R', '9NV': 'V', 'A  ': 'A', 'A1P': 'N', 'A23': 'A',
        'A2L': 'A', 'A2M': 'A', 'A34': 'A', 'A35': 'A', 'A38': 'A',
        'A39': 'A', 'A3A': 'A', 'A3P': 'A', 'A40': 'A', 'A43': 'A',
        'A44': 'A', 'A47': 'A', 'A5L': 'A', 'A5M': 'C', 'A5O': 'A',
        'A66': 'X', 'AA3': 'A', 'AA4': 'A', 'AAR': 'R', 'AB7': 'X',
        'ABA': 'A', 'ABR': 'A', 'ABS': 'A', 'ABT': 'N', 'ACB': 'D',
        'ACL': 'R', 'AD2': 'A', 'ADD': 'X', 'ADX': 'N', 'AEA': 'X',
        'AEI': 'D', 'AET': 'A', 'AFA': 'N', 'AFF': 'N', 'AFG': 'G',
        'AGM': 'R', 'AGT': 'X', 'AHB': 'N', 'AHH': 'X', 'AHO': 'A',
        'AHP': 'A', 'AHS': 'X', 'AHT': 'X', 'AIB': 'A', 'AKL': 'D',
        'ALA': 'A', 'ALC': 'A', 'ALG': 'R', 'ALM': 'A', 'ALN': 'A',
        'ALO': 'T', 'ALQ': 'X', 'ALS': 'A', 'ALT': 'A', 'ALY': 'K',
        'AP7': 'A', 'APE': 'X', 'APH': 'A', 'API': 'K', 'APK': 'K',
        'APM': 'X', 'APP': 'X', 'AR2': 'R', 'AR4': 'E', 'ARG': 'R',
        'ARM': 'R', 'ARO': 'R', 'ARV': 'X', 'AS ': 'A', 'AS2': 'D',
        'AS9': 'X', 'ASA': 'D', 'ASB': 'D', 'ASI': 'D', 'ASK': 'D',
        'ASL': 'D', 'ASM': 'X', 'ASN': 'N', 'ASP': 'D', 'ASQ': 'D',
        'ASU': 'N', 'ASX': 'B', 'ATD': 'T', 'ATL': 'T', 'ATM': 'T',
        'AVC': 'A', 'AVN': 'X', 'AYA': 'A', 'AYG': 'AYG', 'AZK': 'K',
        'AZS': 'S', 'AZY': 'Y', 'B1F': 'F', 'B1P': 'N', 'B2A': 'A',
        'B2F': 'F', 'B2I': 'I', 'B2V': 'V', 'B3A': 'A', 'B3D': 'D',
        'B3E': 'E', 'B3K': 'K', 'B3L': 'X', 'B3M': 'X', 'B3Q': 'X',
        'B3S': 'S', 'B3T': 'X', 'B3U': 'H', 'B3X': 'N', 'B3Y': 'Y',
        'BB6': 'C', 'BB7': 'C', 'BB9': 'C', 'BBC': 'C', 'BCS': 'C',
        'BCX': 'C', 'BE2': 'X', 'BFD': 'D', 'BG1': 'S', 'BGM': 'G',
        'BHD': 'D', 'BIF': 'F', 'BIL': 'X', 'BIU': 'I', 'BJH': 'X',
        'BLE': 'L', 'BLY': 'K', 'BMP': 'N', 'BMT': 'T', 'BNN': 'A',
        'BNO': 'X', 'BOE': 'T', 'BOR': 'R', 'BPE': 'C', 'BRU': 'U',
        'BSE': 'S', 'BT5': 'N', 'BTA': 'L', 'BTC': 'C', 'BTR': 'W',
        'BUC': 'C', 'BUG': 'V', 'BVP': 'U', 'BZG': 'N', 'C  ': 'C',
        'C12': 'TYG', 'C1X': 'K', 'C25': 'C', 'C2L': 'C', 'C2S': 'C',
        'C31': 'C', 'C32': 'C', 'C34': 'C', 'C36': 'C', 'C37': 'C',
        'C38': 'C', 'C3Y': 'C', 'C42': 'C', 'C43': 'C', 'C45': 'C',
        'C46': 'C', 'C49': 'C', 'C4R': 'C', 'C4S': 'C', 'C5C': 'C',
        'C66': 'X', 'C6C': 'C', 'C99': 'TFG', 'CAF': 'C', 'CAL': 'X',
        'CAR': 'C', 'CAS': 'C', 'CAV': 'X', 'CAY': 'C', 'CB2': 'C',
        'CBR': 'C', 'CBV': 'C', 'CCC': 'C', 'CCL': 'K', 'CCS': 'C',
        'CCY': 'CYG', 'CDE': 'X', 'CDV': 'X', 'CDW': 'C', 'CEA': 'C',
        'CFL': 'C', 'CFY': 'FCYG', 'CG1': 'G', 'CGA': 'E', 'CGU': 'E',
        'CH ': 'C', 'CH6': 'MYG', 'CH7': 'KYG', 'CHF': 'X', 'CHG': 'X',
        'CHP': 'G', 'CHS': 'X', 'CIR': 'R', 'CJO': 'GYG', 'CLE': 'L',
        'CLG': 'K', 'CLH': 'K', 'CLV': 'AFG', 'CM0': 'N', 'CME': 'C',
        'CMH': 'C', 'CML': 'C', 'CMR': 'C', 'CMT': 'C', 'CNU': 'U',
        'CP1': 'C', 'CPC': 'X', 'CPI': 'X', 'CQR': 'GYG', 'CR0': 'TLG',
        'CR2': 'GYG', 'CR5': 'G', 'CR7': 'KYG', 'CR8': 'HYG', 'CRF': 'TWG',
        'CRG': 'THG', 'CRK': 'MYG', 'CRO': 'GYG', 'CRQ': 'QYG', 'CRU': 'E',
        'CRW': 'ASG', 'CRX': 'ASG', 'CS0': 'C', 'CS1': 'C', 'CS3': 'C',
        'CS4': 'C', 'CS8': 'N', 'CSA': 'C', 'CSB': 'C', 'CSD': 'C',
        'CSE': 'C', 'CSF': 'C', 'CSH': 'SHG', 'CSI': 'G', 'CSJ': 'C',
        'CSL': 'C', 'CSO': 'C', 'CSP': 'C', 'CSR': 'C', 'CSS': 'C',
        'CSU': 'C', 'CSW': 'C', 'CSX': 'C', 'CSY': 'SYG', 'CSZ': 'C',
        'CTE': 'W', 'CTG': 'T', 'CTH': 'T', 'CUC': 'X', 'CWR': 'S',
        'CXM': 'M', 'CY0': 'C', 'CY1': 'C', 'CY3': 'C', 'CY4': 'C',
        'CYA': 'C', 'CYD': 'C', 'CYF': 'C', 'CYG': 'C', 'CYJ': 'X',
        'CYM': 'C', 'CYQ': 'C', 'CYR': 'C', 'CYS': 'C', 'CZ2': 'C',
        'CZO': 'GYG', 'CZZ': 'C', 'D11': 'T', 'D1P': 'N', 'D3 ': 'N',
        'D33': 'N', 'D3P': 'G', 'D3T': 'T', 'D4M': 'T', 'D4P': 'X',
        'DA ': 'A', 'DA2': 'X', 'DAB': 'A', 'DAH': 'F', 'DAL': 'A',
        'DAR': 'R', 'DAS': 'D', 'DBB': 'T', 'DBM': 'N', 'DBS': 'S',
        'DBU': 'T', 'DBY': 'Y', 'DBZ': 'A', 'DC ': 'C', 'DC2': 'C',
        'DCG': 'G', 'DCI': 'X', 'DCL': 'X', 'DCT': 'C', 'DCY': 'C',
        'DDE': 'H', 'DDG': 'G', 'DDN': 'U', 'DDX': 'N', 'DFC': 'C',
        'DFG': 'G', 'DFI': 'X', 'DFO': 'X', 'DFT': 'N', 'DG ': 'G',
        'DGH': 'G', 'DGI': 'G', 'DGL': 'E', 'DGN': 'Q', 'DHA': 'A',
        'DHI': 'H', 'DHL': 'X', 'DHN': 'V', 'DHP': 'X', 'DHU': 'U',
        'DHV': 'V', 'DI ': 'I', 'DIL': 'I', 'DIR': 'R', 'DIV': 'V',
        'DLE': 'L', 'DLS': 'K', 'DLY': 'K', 'DM0': 'K', 'DMH': 'N',
        'DMK': 'D', 'DMT': 'X', 'DN ': 'N', 'DNE': 'L', 'DNG': 'L',
        'DNL': 'K', 'DNM': 'L', 'DNP': 'A', 'DNR': 'C', 'DNS': 'K',
        'DOA': 'X', 'DOC': 'C', 'DOH': 'D', 'DON': 'L', 'DPB': 'T',
        'DPH': 'F', 'DPL': 'P', 'DPP': 'A', 'DPQ': 'Y', 'DPR': 'P',
        'DPY': 'N', 'DRM': 'U', 'DRP': 'N', 'DRT': 'T', 'DRZ': 'N',
        'DSE': 'S', 'DSG': 'N', 'DSN': 'S', 'DSP': 'D', 'DT ': 'T',
        'DTH': 'T', 'DTR': 'W', 'DTY': 'Y', 'DU ': 'U', 'DVA': 'V',
        'DXD': 'N', 'DXN': 'N', 'DYG': 'DYG', 'DYS': 'C', 'DZM': 'A',
        'E  ': 'A', 'E1X': 'A', 'EDA': 'A', 'EDC': 'G', 'EFC': 'C',
        'EHP': 'F', 'EIT': 'T', 'ENP': 'N', 'ESB': 'Y', 'ESC': 'M',
        'EXY': 'L', 'EY5': 'N', 'EYS': 'X', 'F2F': 'F', 'FA2': 'A',
        'FA5': 'N', 'FAG': 'N', 'FAI': 'N', 'FCL': 'F', 'FFD': 'N',
        'FGL': 'G', 'FGP': 'S', 'FHL': 'X', 'FHO': 'K', 'FHU': 'U',
        'FLA': 'A', 'FLE': 'L', 'FLT': 'Y', 'FME': 'M', 'FMG': 'G',
        'FMU': 'N', 'FOE': 'C', 'FOX': 'G', 'FP9': 'P', 'FPA': 'F',
        'FRD': 'X', 'FT6': 'W', 'FTR': 'W', 'FTY': 'Y', 'FZN': 'K',
        'G  ': 'G', 'G25': 'G', 'G2L': 'G', 'G2S': 'G', 'G31': 'G',
        'G32': 'G', 'G33': 'G', 'G36': 'G', 'G38': 'G', 'G42': 'G',
        'G46': 'G', 'G47': 'G', 'G48': 'G', 'G49': 'G', 'G4P': 'N',
        'G7M': 'G', 'GAO': 'G', 'GAU': 'E', 'GCK': 'C', 'GCM': 'X',
        'GDP': 'G', 'GDR': 'G', 'GFL': 'G', 'GGL': 'E', 'GH3': 'G',
        'GHG': 'Q', 'GHP': 'G', 'GL3': 'G', 'GLH': 'Q', 'GLM': 'X',
        'GLN': 'Q', 'GLQ': 'E', 'GLU': 'E', 'GLX': 'Z', 'GLY': 'G',
        'GLZ': 'G', 'GMA': 'E', 'GMS': 'G', 'GMU': 'U', 'GN7': 'G',
        'GND': 'X', 'GNE': 'N', 'GOM': 'G', 'GPL': 'K', 'GS ': 'G',
        'GSC': 'G', 'GSR': 'G', 'GSS': 'G', 'GSU': 'E', 'GT9': 'C',
        'GTP': 'G', 'GVL': 'X', 'GYC': 'CYG', 'GYS': 'SYG', 'H2U': 'U',
        'H5M': 'P', 'HAC': 'A', 'HAR': 'R', 'HBN': 'H', 'HCS': 'X',
        'HDP': 'U', 'HEU': 'U', 'HFA': 'X', 'HGL': 'X', 'HHI': 'H',
        'HHK': 'AK', 'HIA': 'H', 'HIC': 'H', 'HIP': 'H', 'HIQ': 'H',
        'HIS': 'H', 'HL2': 'L', 'HLU': 'L', 'HMF': 'A', 'HMR': 'R',
        'HOL': 'N', 'HPC': 'F', 'HPE': 'F', 'HPQ': 'F', 'HQA': 'A',
        'HRG': 'R', 'HRP': 'W', 'HS8': 'H', 'HS9': 'H', 'HSE': 'S',
        'HSL': 'S', 'HSO': 'H', 'HTI': 'C', 'HTN': 'N', 'HTR': 'W',
        'HV5': 'A', 'HVA': 'V', 'HY3': 'P', 'HYP': 'P', 'HZP': 'P',
        'I  ': 'I', 'I2M': 'I', 'I58': 'K', 'I5C': 'C', 'IAM': 'A',
        'IAR': 'R', 'IAS': 'D', 'IC ': 'C', 'IEL': 'K', 'IEY': 'HYG',
        'IG ': 'G', 'IGL': 'G', 'IGU': 'G', 'IIC': 'SHG', 'IIL': 'I',
        'ILE': 'I', 'ILG': 'E', 'ILX': 'I', 'IMC': 'C', 'IML': 'I',
        'IOY': 'F', 'IPG': 'G', 'IPN': 'N', 'IRN': 'N', 'IT1': 'K',
        'IU ': 'U', 'IYR': 'Y', 'IYT': 'T', 'JJJ': 'C', 'JJK': 'C',
        'JJL': 'C', 'JW5': 'N', 'K1R': 'C', 'KAG': 'G', 'KCX': 'K',
        'KGC': 'K', 'KOR': 'M', 'KPI': 'K', 'KST': 'K', 'KYQ': 'K',
        'L2A': 'X', 'LA2': 'K', 'LAA': 'D', 'LAL': 'A', 'LBY': 'K',
        'LC ': 'C', 'LCA': 'A', 'LCC': 'N', 'LCG': 'G', 'LCH': 'N',
        'LCK': 'K', 'LCX': 'K', 'LDH': 'K', 'LED': 'L', 'LEF': 'L',
        'LEH': 'L', 'LEI': 'V', 'LEM': 'L', 'LEN': 'L', 'LET': 'X',
        'LEU': 'L', 'LG ': 'G', 'LGP': 'G', 'LHC': 'X', 'LHU': 'U',
        'LKC': 'N', 'LLP': 'K', 'LLY': 'K', 'LME': 'E', 'LMQ': 'Q',
        'LMS': 'N', 'LP6': 'K', 'LPD': 'P', 'LPG': 'G', 'LPL': 'X',
        'LPS': 'S', 'LSO': 'X', 'LTA': 'X', 'LTR': 'W', 'LVG': 'G',
        'LVN': 'V', 'LYM': 'K', 'LYN': 'K', 'LYR': 'K', 'LYS': 'K',
        'LYX': 'K', 'LYZ': 'K', 'M0H': 'C', 'M1G': 'G', 'M2G': 'G',
        'M2L': 'K', 'M2S': 'M', 'M3L': 'K', 'M5M': 'C', 'MA ': 'A',
        'MA6': 'A', 'MA7': 'A', 'MAA': 'A', 'MAD': 'A', 'MAI': 'R',
        'MBQ': 'Y', 'MBZ': 'N', 'MC1': 'S', 'MCG': 'X', 'MCL': 'K',
        'MCS': 'C', 'MCY': 'C', 'MDH': 'X', 'MDO': 'ASG', 'MDR': 'N',
        'MEA': 'F', 'MED': 'M', 'MEG': 'E', 'MEN': 'N', 'MEP': 'U',
        'MEQ': 'Q', 'MET': 'M', 'MEU': 'G', 'MF3': 'X', 'MFC': 'GYG',
        'MG1': 'G', 'MGG': 'R', 'MGN': 'Q', 'MGQ': 'A', 'MGV': 'G',
        'MGY': 'G', 'MHL': 'L', 'MHO': 'M', 'MHS': 'H', 'MIA': 'A',
        'MIS': 'S', 'MK8': 'L', 'ML3': 'K', 'MLE': 'L', 'MLL': 'L',
        'MLY': 'K', 'MLZ': 'K', 'MME': 'M', 'MMT': 'T', 'MND': 'N',
        'MNL': 'L', 'MNU': 'U', 'MNV': 'V', 'MOD': 'X', 'MP8': 'P',
        'MPH': 'X', 'MPJ': 'X', 'MPQ': 'G', 'MRG': 'G', 'MSA': 'G',
        'MSE': 'M', 'MSL': 'M', 'MSO': 'M', 'MSP': 'X', 'MT2': 'M',
        'MTR': 'T', 'MTU': 'A', 'MTY': 'Y', 'MVA': 'V', 'N  ': 'N',
        'N10': 'S', 'N2C': 'X', 'N5I': 'N', 'N5M': 'C', 'N6G': 'G',
        'N7P': 'P', 'NA8': 'A', 'NAL': 'A', 'NAM': 'A', 'NB8': 'N',
        'NBQ': 'Y', 'NC1': 'S', 'NCB': 'A', 'NCX': 'N', 'NCY': 'X',
        'NDF': 'F', 'NDN': 'U', 'NEM': 'H', 'NEP': 'H', 'NF2': 'N',
        'NFA': 'F', 'NHL': 'E', 'NIT': 'X', 'NIY': 'Y', 'NLE': 'L',
        'NLN': 'L', 'NLO': 'L', 'NLP': 'L', 'NLQ': 'Q', 'NMC': 'G',
        'NMM': 'R', 'NMS': 'T', 'NMT': 'T', 'NNH': 'R', 'NP3': 'N',
        'NPH': 'C', 'NRP': 'LYG', 'NRQ': 'MYG', 'NSK': 'X', 'NTY': 'Y',
        'NVA': 'V', 'NYC': 'TWG', 'NYG': 'NYG', 'NYM': 'N', 'NYS': 'C',
        'NZH': 'H', 'O12': 'X', 'O2C': 'N', 'O2G': 'G', 'OAD': 'N',
        'OAS': 'S', 'OBF': 'X', 'OBS': 'X', 'OCS': 'C', 'OCY': 'C',
        'ODP': 'N', 'OHI': 'H', 'OHS': 'D', 'OIC': 'X', 'OIP': 'I',
        'OLE': 'X', 'OLT': 'T', 'OLZ': 'S', 'OMC': 'C', 'OMG': 'G',
        'OMT': 'M', 'OMU': 'U', 'ONE': 'U', 'ONL': 'X', 'OPR': 'R',
        'ORN': 'A', 'ORQ': 'R', 'OSE': 'S', 'OTB': 'X', 'OTH': 'T',
        'OTY': 'Y', 'OXX': 'D', 'P  ': 'G', 'P1L': 'C', 'P1P': 'N',
        'P2T': 'T', 'P2U': 'U', 'P2Y': 'P', 'P5P': 'A', 'PAQ': 'Y',
        'PAS': 'D', 'PAT': 'W', 'PAU': 'A', 'PBB': 'C', 'PBF': 'F',
        'PBT': 'N', 'PCA': 'E', 'PCC': 'P', 'PCE': 'X', 'PCS': 'F',
        'PDL': 'X', 'PDU': 'U', 'PEC': 'C', 'PF5': 'F', 'PFF': 'F',
        'PFX': 'X', 'PG1': 'S', 'PG7': 'G', 'PG9': 'G', 'PGL': 'X',
        'PGN': 'G', 'PGP': 'G', 'PGY': 'G', 'PHA': 'F', 'PHD': 'D',
        'PHE': 'F', 'PHI': 'F', 'PHL': 'F', 'PHM': 'F', 'PIV': 'X',
        'PLE': 'L', 'PM3': 'F', 'PMT': 'C', 'POM': 'P', 'PPN': 'F',
        'PPU': 'A', 'PPW': 'G', 'PQ1': 'N', 'PR3': 'C', 'PR5': 'A',
        'PR9': 'P', 'PRN': 'A', 'PRO': 'P', 'PRS': 'P', 'PSA': 'F',
        'PSH': 'H', 'PST': 'T', 'PSU': 'U', 'PSW': 'C', 'PTA': 'X',
        'PTH': 'Y', 'PTM': 'Y', 'PTR': 'Y', 'PU ': 'A', 'PUY': 'N',
        'PVH': 'H', 'PVL': 'X', 'PYA': 'A', 'PYO': 'U', 'PYX': 'C',
        'PYY': 'N', 'QLG': 'QLG', 'QUO': 'G', 'R  ': 'A', 'R1A': 'C',
        'R1B': 'C', 'R1F': 'C', 'R7A': 'C', 'RC7': 'HYG', 'RCY': 'C',
        'RIA': 'A', 'RMP': 'A', 'RON': 'X', 'RT ': 'T', 'RTP': 'N',
        'S1H': 'S', 'S2C': 'C', 'S2D': 'A', 'S2M': 'T', 'S2P': 'A',
        'S4A': 'A', 'S4C': 'C', 'S4G': 'G', 'S4U': 'U', 'S6G': 'G',
        'SAC': 'S', 'SAH': 'C', 'SAR': 'G', 'SBL': 'S', 'SC ': 'C',
        'SCH': 'C', 'SCS': 'C', 'SCY': 'C', 'SD2': 'X', 'SDG': 'G',
        'SDP': 'S', 'SEB': 'S', 'SEC': 'A', 'SEG': 'A', 'SEL': 'S',
        'SEM': 'X', 'SEN': 'S', 'SEP': 'S', 'SER': 'S', 'SET': 'S',
        'SGB': 'S', 'SHC': 'C', 'SHP': 'G', 'SHR': 'K', 'SIB': 'C',
        'SIC': 'DC', 'SLA': 'P', 'SLR': 'P', 'SLZ': 'K', 'SMC': 'C',
        'SME': 'M', 'SMF': 'F', 'SMP': 'A', 'SMT': 'T', 'SNC': 'C',
        'SNN': 'N', 'SOC': 'C', 'SOS': 'N', 'SOY': 'S', 'SPT': 'T',
        'SRA': 'A', 'SSU': 'U', 'STY': 'Y', 'SUB': 'X', 'SUI': 'DG',
        'SUN': 'S', 'SUR': 'U', 'SVA': 'S', 'SVX': 'S', 'SVZ': 'X',
        'SYS': 'C', 'T  ': 'T', 'T11': 'F', 'T23': 'T', 'T2S': 'T',
        'T2T': 'N', 'T31': 'U', 'T32': 'T', 'T36': 'T', 'T37': 'T',
        'T38': 'T', 'T39': 'T', 'T3P': 'T', 'T41': 'T', 'T48': 'T',
        'T49': 'T', 'T4S': 'T', 'T5O': 'U', 'T5S': 'T', 'T66': 'X',
        'T6A': 'A', 'TA3': 'T', 'TA4': 'X', 'TAF': 'T', 'TAL': 'N',
        'TAV': 'D', 'TBG': 'V', 'TBM': 'T', 'TC1': 'C', 'TCP': 'T',
        'TCQ': 'X', 'TCR': 'W', 'TCY': 'A', 'TDD': 'L', 'TDY': 'T',
        'TFE': 'T', 'TFO': 'A', 'TFQ': 'F', 'TFT': 'T', 'TGP': 'G',
        'TH6': 'T', 'THC': 'T', 'THO': 'X', 'THR': 'T', 'THX': 'N',
        'THZ': 'R', 'TIH': 'A', 'TLB': 'N', 'TLC': 'T', 'TLN': 'U',
        'TMB': 'T', 'TMD': 'T', 'TNB': 'C', 'TNR': 'S', 'TOX': 'W',
        'TP1': 'T', 'TPC': 'C', 'TPG': 'G', 'TPH': 'X', 'TPL': 'W',
        'TPO': 'T', 'TPQ': 'Y', 'TQQ': 'W', 'TRF': 'W', 'TRG': 'K',
        'TRN': 'W', 'TRO': 'W', 'TRP': 'W', 'TRQ': 'W', 'TRW': 'W',
        'TRX': 'W', 'TS ': 'N', 'TST': 'X', 'TT ': 'N', 'TTD': 'T',
        'TTI': 'U', 'TTM': 'T', 'TTQ': 'W', 'TTS': 'Y', 'TY2': 'Y',
        'TY3': 'Y', 'TYB': 'Y', 'TYI': 'Y', 'TYN': 'Y', 'TYO': 'Y',
        'TYQ': 'Y', 'TYR': 'Y', 'TYS': 'Y', 'TYT': 'Y', 'TYU': 'N',
        'TYX': 'X', 'TYY': 'Y', 'TZB': 'X', 'TZO': 'X', 'U  ': 'U',
        'U25': 'U', 'U2L': 'U', 'U2N': 'U', 'U2P': 'U', 'U31': 'U',
        'U33': 'U', 'U34': 'U', 'U36': 'U', 'U37': 'U', 'U8U': 'U',
        'UAR': 'U', 'UCL': 'U', 'UD5': 'U', 'UDP': 'N', 'UFP': 'N',
        'UFR': 'U', 'UFT': 'U', 'UMA': 'A', 'UMP': 'U', 'UMS': 'U',
        'UN1': 'X', 'UN2': 'X', 'UNK': 'X', 'UR3': 'U', 'URD': 'U',
        'US1': 'U', 'US2': 'U', 'US3': 'T', 'US5': 'U', 'USM': 'U',
        'V1A': 'C', 'VAD': 'V', 'VAF': 'V', 'VAL': 'V', 'VB1': 'K',
        'VDL': 'X', 'VLL': 'X', 'VLM': 'X', 'VMS': 'X', 'VOL': 'X',
        'X  ': 'G', 'X2W': 'E', 'X4A': 'N', 'X9Q': 'AFG', 'XAD': 'A',
        'XAE': 'N', 'XAL': 'A', 'XAR': 'N', 'XCL': 'C', 'XCP': 'X',
        'XCR': 'C', 'XCS': 'N', 'XCT': 'C', 'XCY': 'C', 'XGA': 'N',
        'XGL': 'G', 'XGR': 'G', 'XGU': 'G', 'XTH': 'T', 'XTL': 'T',
        'XTR': 'T', 'XTS': 'G', 'XTY': 'N', 'XUA': 'A', 'XUG': 'G',
        'XX1': 'K', 'XXY': 'THG', 'XYG': 'DYG', 'Y  ': 'A', 'YCM': 'C',
        'YG ': 'G', 'YOF': 'Y', 'YRR': 'N', 'YYG': 'G', 'Z  ': 'C',
        'ZAD': 'A', 'ZAL': 'A', 'ZBC': 'C', 'ZCY': 'C', 'ZDU': 'U',
        'ZFB': 'X', 'ZGU': 'G', 'ZHP': 'N', 'ZTH': 'T', 'ZZJ': 'A'}

    # remove hetero atoms (waters/ligands/etc) from consideration?


    cmd.select("__h", "br. " + selName + " and not het")

    # get the AAs in the haystack
    aaDict = {'aaList': []}
    cmd.iterate("(name ca) and __h", "aaList.append((resi,resn,chain))", space=aaDict)

    IDs = [int(x[0]) for x in aaDict['aaList']]
    AAs = ''.join([one_letter[x[1]] for x in aaDict['aaList']])
    return AAs,IDs
def find_amino_in_selection(selName=None, het=0, firstOnly=0):

    one_letter = {
        '00C': 'C', '01W': 'X', '0A0': 'D', '0A1': 'Y', '0A2': 'K',
        '0A8': 'C', '0AA': 'V', '0AB': 'V', '0AC': 'G', '0AD': 'G',
        '0AF': 'W', '0AG': 'L', '0AH': 'S', '0AK': 'D', '0AM': 'A',
        '0AP': 'C', '0AU': 'U', '0AV': 'A', '0AZ': 'P', '0BN': 'F',
        '0C ': 'C', '0CS': 'A', '0DC': 'C', '0DG': 'G', '0DT': 'T',
        '0G ': 'G', '0NC': 'A', '0SP': 'A', '0U ': 'U', '0YG': 'YG',
        '10C': 'C', '125': 'U', '126': 'U', '127': 'U', '128': 'N',
        '12A': 'A', '143': 'C', '175': 'ASG', '193': 'X', '1AP': 'A',
        '1MA': 'A', '1MG': 'G', '1PA': 'F', '1PI': 'A', '1PR': 'N',
        '1SC': 'C', '1TQ': 'W', '1TY': 'Y', '200': 'F', '23F': 'F',
        '23S': 'X', '26B': 'T', '2AD': 'X', '2AG': 'G', '2AO': 'X',
        '2AR': 'A', '2AS': 'X', '2AT': 'T', '2AU': 'U', '2BD': 'I',
        '2BT': 'T', '2BU': 'A', '2CO': 'C', '2DA': 'A', '2DF': 'N',
        '2DM': 'N', '2DO': 'X', '2DT': 'T', '2EG': 'G', '2FE': 'N',
        '2FI': 'N', '2FM': 'M', '2GT': 'T', '2HF': 'H', '2LU': 'L',
        '2MA': 'A', '2MG': 'G', '2ML': 'L', '2MR': 'R', '2MT': 'P',
        '2MU': 'U', '2NT': 'T', '2OM': 'U', '2OT': 'T', '2PI': 'X',
        '2PR': 'G', '2SA': 'N', '2SI': 'X', '2ST': 'T', '2TL': 'T',
        '2TY': 'Y', '2VA': 'V', '32S': 'X', '32T': 'X', '3AH': 'H',
        '3AR': 'X', '3CF': 'F', '3DA': 'A', '3DR': 'N', '3GA': 'A',
        '3MD': 'D', '3ME': 'U', '3NF': 'Y', '3TY': 'X', '3XH': 'G',
        '4AC': 'N', '4BF': 'Y', '4CF': 'F', '4CY': 'M', '4DP': 'W',
        '4F3': 'GYG', '4FB': 'P', '4FW': 'W', '4HT': 'W', '4IN': 'X',
        '4MF': 'N', '4MM': 'X', '4OC': 'C', '4PC': 'C', '4PD': 'C',
        '4PE': 'C', '4PH': 'F', '4SC': 'C', '4SU': 'U', '4TA': 'N',
        '5AA': 'A', '5AT': 'T', '5BU': 'U', '5CG': 'G', '5CM': 'C',
        '5CS': 'C', '5FA': 'A', '5FC': 'C', '5FU': 'U', '5HP': 'E',
        '5HT': 'T', '5HU': 'U', '5IC': 'C', '5IT': 'T', '5IU': 'U',
        '5MC': 'C', '5MD': 'N', '5MU': 'U', '5NC': 'C', '5PC': 'C',
        '5PY': 'T', '5SE': 'U', '5ZA': 'TWG', '64T': 'T', '6CL': 'K',
        '6CT': 'T', '6CW': 'W', '6HA': 'A', '6HC': 'C', '6HG': 'G',
        '6HN': 'K', '6HT': 'T', '6IA': 'A', '6MA': 'A', '6MC': 'A',
        '6MI': 'N', '6MT': 'A', '6MZ': 'N', '6OG': 'G', '70U': 'U',
        '7DA': 'A', '7GU': 'G', '7JA': 'I', '7MG': 'G', '8AN': 'A',
        '8FG': 'G', '8MG': 'G', '8OG': 'G', '9NE': 'E', '9NF': 'F',
        '9NR': 'R', '9NV': 'V', 'A  ': 'A', 'A1P': 'N', 'A23': 'A',
        'A2L': 'A', 'A2M': 'A', 'A34': 'A', 'A35': 'A', 'A38': 'A',
        'A39': 'A', 'A3A': 'A', 'A3P': 'A', 'A40': 'A', 'A43': 'A',
        'A44': 'A', 'A47': 'A', 'A5L': 'A', 'A5M': 'C', 'A5O': 'A',
        'A66': 'X', 'AA3': 'A', 'AA4': 'A', 'AAR': 'R', 'AB7': 'X',
        'ABA': 'A', 'ABR': 'A', 'ABS': 'A', 'ABT': 'N', 'ACB': 'D',
        'ACL': 'R', 'AD2': 'A', 'ADD': 'X', 'ADX': 'N', 'AEA': 'X',
        'AEI': 'D', 'AET': 'A', 'AFA': 'N', 'AFF': 'N', 'AFG': 'G',
        'AGM': 'R', 'AGT': 'X', 'AHB': 'N', 'AHH': 'X', 'AHO': 'A',
        'AHP': 'A', 'AHS': 'X', 'AHT': 'X', 'AIB': 'A', 'AKL': 'D',
        'ALA': 'A', 'ALC': 'A', 'ALG': 'R', 'ALM': 'A', 'ALN': 'A',
        'ALO': 'T', 'ALQ': 'X', 'ALS': 'A', 'ALT': 'A', 'ALY': 'K',
        'AP7': 'A', 'APE': 'X', 'APH': 'A', 'API': 'K', 'APK': 'K',
        'APM': 'X', 'APP': 'X', 'AR2': 'R', 'AR4': 'E', 'ARG': 'R',
        'ARM': 'R', 'ARO': 'R', 'ARV': 'X', 'AS ': 'A', 'AS2': 'D',
        'AS9': 'X', 'ASA': 'D', 'ASB': 'D', 'ASI': 'D', 'ASK': 'D',
        'ASL': 'D', 'ASM': 'X', 'ASN': 'N', 'ASP': 'D', 'ASQ': 'D',
        'ASU': 'N', 'ASX': 'B', 'ATD': 'T', 'ATL': 'T', 'ATM': 'T',
        'AVC': 'A', 'AVN': 'X', 'AYA': 'A', 'AYG': 'AYG', 'AZK': 'K',
        'AZS': 'S', 'AZY': 'Y', 'B1F': 'F', 'B1P': 'N', 'B2A': 'A',
        'B2F': 'F', 'B2I': 'I', 'B2V': 'V', 'B3A': 'A', 'B3D': 'D',
        'B3E': 'E', 'B3K': 'K', 'B3L': 'X', 'B3M': 'X', 'B3Q': 'X',
        'B3S': 'S', 'B3T': 'X', 'B3U': 'H', 'B3X': 'N', 'B3Y': 'Y',
        'BB6': 'C', 'BB7': 'C', 'BB9': 'C', 'BBC': 'C', 'BCS': 'C',
        'BCX': 'C', 'BE2': 'X', 'BFD': 'D', 'BG1': 'S', 'BGM': 'G',
        'BHD': 'D', 'BIF': 'F', 'BIL': 'X', 'BIU': 'I', 'BJH': 'X',
        'BLE': 'L', 'BLY': 'K', 'BMP': 'N', 'BMT': 'T', 'BNN': 'A',
        'BNO': 'X', 'BOE': 'T', 'BOR': 'R', 'BPE': 'C', 'BRU': 'U',
        'BSE': 'S', 'BT5': 'N', 'BTA': 'L', 'BTC': 'C', 'BTR': 'W',
        'BUC': 'C', 'BUG': 'V', 'BVP': 'U', 'BZG': 'N', 'C  ': 'C',
        'C12': 'TYG', 'C1X': 'K', 'C25': 'C', 'C2L': 'C', 'C2S': 'C',
        'C31': 'C', 'C32': 'C', 'C34': 'C', 'C36': 'C', 'C37': 'C',
        'C38': 'C', 'C3Y': 'C', 'C42': 'C', 'C43': 'C', 'C45': 'C',
        'C46': 'C', 'C49': 'C', 'C4R': 'C', 'C4S': 'C', 'C5C': 'C',
        'C66': 'X', 'C6C': 'C', 'C99': 'TFG', 'CAF': 'C', 'CAL': 'X',
        'CAR': 'C', 'CAS': 'C', 'CAV': 'X', 'CAY': 'C', 'CB2': 'C',
        'CBR': 'C', 'CBV': 'C', 'CCC': 'C', 'CCL': 'K', 'CCS': 'C',
        'CCY': 'CYG', 'CDE': 'X', 'CDV': 'X', 'CDW': 'C', 'CEA': 'C',
        'CFL': 'C', 'CFY': 'FCYG', 'CG1': 'G', 'CGA': 'E', 'CGU': 'E',
        'CH ': 'C', 'CH6': 'MYG', 'CH7': 'KYG', 'CHF': 'X', 'CHG': 'X',
        'CHP': 'G', 'CHS': 'X', 'CIR': 'R', 'CJO': 'GYG', 'CLE': 'L',
        'CLG': 'K', 'CLH': 'K', 'CLV': 'AFG', 'CM0': 'N', 'CME': 'C',
        'CMH': 'C', 'CML': 'C', 'CMR': 'C', 'CMT': 'C', 'CNU': 'U',
        'CP1': 'C', 'CPC': 'X', 'CPI': 'X', 'CQR': 'GYG', 'CR0': 'TLG',
        'CR2': 'GYG', 'CR5': 'G', 'CR7': 'KYG', 'CR8': 'HYG', 'CRF': 'TWG',
        'CRG': 'THG', 'CRK': 'MYG', 'CRO': 'GYG', 'CRQ': 'QYG', 'CRU': 'E',
        'CRW': 'ASG', 'CRX': 'ASG', 'CS0': 'C', 'CS1': 'C', 'CS3': 'C',
        'CS4': 'C', 'CS8': 'N', 'CSA': 'C', 'CSB': 'C', 'CSD': 'C',
        'CSE': 'C', 'CSF': 'C', 'CSH': 'SHG', 'CSI': 'G', 'CSJ': 'C',
        'CSL': 'C', 'CSO': 'C', 'CSP': 'C', 'CSR': 'C', 'CSS': 'C',
        'CSU': 'C', 'CSW': 'C', 'CSX': 'C', 'CSY': 'SYG', 'CSZ': 'C',
        'CTE': 'W', 'CTG': 'T', 'CTH': 'T', 'CUC': 'X', 'CWR': 'S',
        'CXM': 'M', 'CY0': 'C', 'CY1': 'C', 'CY3': 'C', 'CY4': 'C',
        'CYA': 'C', 'CYD': 'C', 'CYF': 'C', 'CYG': 'C', 'CYJ': 'X',
        'CYM': 'C', 'CYQ': 'C', 'CYR': 'C', 'CYS': 'C', 'CZ2': 'C',
        'CZO': 'GYG', 'CZZ': 'C', 'D11': 'T', 'D1P': 'N', 'D3 ': 'N',
        'D33': 'N', 'D3P': 'G', 'D3T': 'T', 'D4M': 'T', 'D4P': 'X',
        'DA ': 'A', 'DA2': 'X', 'DAB': 'A', 'DAH': 'F', 'DAL': 'A',
        'DAR': 'R', 'DAS': 'D', 'DBB': 'T', 'DBM': 'N', 'DBS': 'S',
        'DBU': 'T', 'DBY': 'Y', 'DBZ': 'A', 'DC ': 'C', 'DC2': 'C',
        'DCG': 'G', 'DCI': 'X', 'DCL': 'X', 'DCT': 'C', 'DCY': 'C',
        'DDE': 'H', 'DDG': 'G', 'DDN': 'U', 'DDX': 'N', 'DFC': 'C',
        'DFG': 'G', 'DFI': 'X', 'DFO': 'X', 'DFT': 'N', 'DG ': 'G',
        'DGH': 'G', 'DGI': 'G', 'DGL': 'E', 'DGN': 'Q', 'DHA': 'A',
        'DHI': 'H', 'DHL': 'X', 'DHN': 'V', 'DHP': 'X', 'DHU': 'U',
        'DHV': 'V', 'DI ': 'I', 'DIL': 'I', 'DIR': 'R', 'DIV': 'V',
        'DLE': 'L', 'DLS': 'K', 'DLY': 'K', 'DM0': 'K', 'DMH': 'N',
        'DMK': 'D', 'DMT': 'X', 'DN ': 'N', 'DNE': 'L', 'DNG': 'L',
        'DNL': 'K', 'DNM': 'L', 'DNP': 'A', 'DNR': 'C', 'DNS': 'K',
        'DOA': 'X', 'DOC': 'C', 'DOH': 'D', 'DON': 'L', 'DPB': 'T',
        'DPH': 'F', 'DPL': 'P', 'DPP': 'A', 'DPQ': 'Y', 'DPR': 'P',
        'DPY': 'N', 'DRM': 'U', 'DRP': 'N', 'DRT': 'T', 'DRZ': 'N',
        'DSE': 'S', 'DSG': 'N', 'DSN': 'S', 'DSP': 'D', 'DT ': 'T',
        'DTH': 'T', 'DTR': 'W', 'DTY': 'Y', 'DU ': 'U', 'DVA': 'V',
        'DXD': 'N', 'DXN': 'N', 'DYG': 'DYG', 'DYS': 'C', 'DZM': 'A',
        'E  ': 'A', 'E1X': 'A', 'EDA': 'A', 'EDC': 'G', 'EFC': 'C',
        'EHP': 'F', 'EIT': 'T', 'ENP': 'N', 'ESB': 'Y', 'ESC': 'M',
        'EXY': 'L', 'EY5': 'N', 'EYS': 'X', 'F2F': 'F', 'FA2': 'A',
        'FA5': 'N', 'FAG': 'N', 'FAI': 'N', 'FCL': 'F', 'FFD': 'N',
        'FGL': 'G', 'FGP': 'S', 'FHL': 'X', 'FHO': 'K', 'FHU': 'U',
        'FLA': 'A', 'FLE': 'L', 'FLT': 'Y', 'FME': 'M', 'FMG': 'G',
        'FMU': 'N', 'FOE': 'C', 'FOX': 'G', 'FP9': 'P', 'FPA': 'F',
        'FRD': 'X', 'FT6': 'W', 'FTR': 'W', 'FTY': 'Y', 'FZN': 'K',
        'G  ': 'G', 'G25': 'G', 'G2L': 'G', 'G2S': 'G', 'G31': 'G',
        'G32': 'G', 'G33': 'G', 'G36': 'G', 'G38': 'G', 'G42': 'G',
        'G46': 'G', 'G47': 'G', 'G48': 'G', 'G49': 'G', 'G4P': 'N',
        'G7M': 'G', 'GAO': 'G', 'GAU': 'E', 'GCK': 'C', 'GCM': 'X',
        'GDP': 'G', 'GDR': 'G', 'GFL': 'G', 'GGL': 'E', 'GH3': 'G',
        'GHG': 'Q', 'GHP': 'G', 'GL3': 'G', 'GLH': 'Q', 'GLM': 'X',
        'GLN': 'Q', 'GLQ': 'E', 'GLU': 'E', 'GLX': 'Z', 'GLY': 'G',
        'GLZ': 'G', 'GMA': 'E', 'GMS': 'G', 'GMU': 'U', 'GN7': 'G',
        'GND': 'X', 'GNE': 'N', 'GOM': 'G', 'GPL': 'K', 'GS ': 'G',
        'GSC': 'G', 'GSR': 'G', 'GSS': 'G', 'GSU': 'E', 'GT9': 'C',
        'GTP': 'G', 'GVL': 'X', 'GYC': 'CYG', 'GYS': 'SYG', 'H2U': 'U',
        'H5M': 'P', 'HAC': 'A', 'HAR': 'R', 'HBN': 'H', 'HCS': 'X',
        'HDP': 'U', 'HEU': 'U', 'HFA': 'X', 'HGL': 'X', 'HHI': 'H',
        'HHK': 'AK', 'HIA': 'H', 'HIC': 'H', 'HIP': 'H', 'HIQ': 'H',
        'HIS': 'H', 'HL2': 'L', 'HLU': 'L', 'HMF': 'A', 'HMR': 'R',
        'HOL': 'N', 'HPC': 'F', 'HPE': 'F', 'HPQ': 'F', 'HQA': 'A',
        'HRG': 'R', 'HRP': 'W', 'HS8': 'H', 'HS9': 'H', 'HSE': 'S',
        'HSL': 'S', 'HSO': 'H', 'HTI': 'C', 'HTN': 'N', 'HTR': 'W',
        'HV5': 'A', 'HVA': 'V', 'HY3': 'P', 'HYP': 'P', 'HZP': 'P',
        'I  ': 'I', 'I2M': 'I', 'I58': 'K', 'I5C': 'C', 'IAM': 'A',
        'IAR': 'R', 'IAS': 'D', 'IC ': 'C', 'IEL': 'K', 'IEY': 'HYG',
        'IG ': 'G', 'IGL': 'G', 'IGU': 'G', 'IIC': 'SHG', 'IIL': 'I',
        'ILE': 'I', 'ILG': 'E', 'ILX': 'I', 'IMC': 'C', 'IML': 'I',
        'IOY': 'F', 'IPG': 'G', 'IPN': 'N', 'IRN': 'N', 'IT1': 'K',
        'IU ': 'U', 'IYR': 'Y', 'IYT': 'T', 'JJJ': 'C', 'JJK': 'C',
        'JJL': 'C', 'JW5': 'N', 'K1R': 'C', 'KAG': 'G', 'KCX': 'K',
        'KGC': 'K', 'KOR': 'M', 'KPI': 'K', 'KST': 'K', 'KYQ': 'K',
        'L2A': 'X', 'LA2': 'K', 'LAA': 'D', 'LAL': 'A', 'LBY': 'K',
        'LC ': 'C', 'LCA': 'A', 'LCC': 'N', 'LCG': 'G', 'LCH': 'N',
        'LCK': 'K', 'LCX': 'K', 'LDH': 'K', 'LED': 'L', 'LEF': 'L',
        'LEH': 'L', 'LEI': 'V', 'LEM': 'L', 'LEN': 'L', 'LET': 'X',
        'LEU': 'L', 'LG ': 'G', 'LGP': 'G', 'LHC': 'X', 'LHU': 'U',
        'LKC': 'N', 'LLP': 'K', 'LLY': 'K', 'LME': 'E', 'LMQ': 'Q',
        'LMS': 'N', 'LP6': 'K', 'LPD': 'P', 'LPG': 'G', 'LPL': 'X',
        'LPS': 'S', 'LSO': 'X', 'LTA': 'X', 'LTR': 'W', 'LVG': 'G',
        'LVN': 'V', 'LYM': 'K', 'LYN': 'K', 'LYR': 'K', 'LYS': 'K',
        'LYX': 'K', 'LYZ': 'K', 'M0H': 'C', 'M1G': 'G', 'M2G': 'G',
        'M2L': 'K', 'M2S': 'M', 'M3L': 'K', 'M5M': 'C', 'MA ': 'A',
        'MA6': 'A', 'MA7': 'A', 'MAA': 'A', 'MAD': 'A', 'MAI': 'R',
        'MBQ': 'Y', 'MBZ': 'N', 'MC1': 'S', 'MCG': 'X', 'MCL': 'K',
        'MCS': 'C', 'MCY': 'C', 'MDH': 'X', 'MDO': 'ASG', 'MDR': 'N',
        'MEA': 'F', 'MED': 'M', 'MEG': 'E', 'MEN': 'N', 'MEP': 'U',
        'MEQ': 'Q', 'MET': 'M', 'MEU': 'G', 'MF3': 'X', 'MFC': 'GYG',
        'MG1': 'G', 'MGG': 'R', 'MGN': 'Q', 'MGQ': 'A', 'MGV': 'G',
        'MGY': 'G', 'MHL': 'L', 'MHO': 'M', 'MHS': 'H', 'MIA': 'A',
        'MIS': 'S', 'MK8': 'L', 'ML3': 'K', 'MLE': 'L', 'MLL': 'L',
        'MLY': 'K', 'MLZ': 'K', 'MME': 'M', 'MMT': 'T', 'MND': 'N',
        'MNL': 'L', 'MNU': 'U', 'MNV': 'V', 'MOD': 'X', 'MP8': 'P',
        'MPH': 'X', 'MPJ': 'X', 'MPQ': 'G', 'MRG': 'G', 'MSA': 'G',
        'MSE': 'M', 'MSL': 'M', 'MSO': 'M', 'MSP': 'X', 'MT2': 'M',
        'MTR': 'T', 'MTU': 'A', 'MTY': 'Y', 'MVA': 'V', 'N  ': 'N',
        'N10': 'S', 'N2C': 'X', 'N5I': 'N', 'N5M': 'C', 'N6G': 'G',
        'N7P': 'P', 'NA8': 'A', 'NAL': 'A', 'NAM': 'A', 'NB8': 'N',
        'NBQ': 'Y', 'NC1': 'S', 'NCB': 'A', 'NCX': 'N', 'NCY': 'X',
        'NDF': 'F', 'NDN': 'U', 'NEM': 'H', 'NEP': 'H', 'NF2': 'N',
        'NFA': 'F', 'NHL': 'E', 'NIT': 'X', 'NIY': 'Y', 'NLE': 'L',
        'NLN': 'L', 'NLO': 'L', 'NLP': 'L', 'NLQ': 'Q', 'NMC': 'G',
        'NMM': 'R', 'NMS': 'T', 'NMT': 'T', 'NNH': 'R', 'NP3': 'N',
        'NPH': 'C', 'NRP': 'LYG', 'NRQ': 'MYG', 'NSK': 'X', 'NTY': 'Y',
        'NVA': 'V', 'NYC': 'TWG', 'NYG': 'NYG', 'NYM': 'N', 'NYS': 'C',
        'NZH': 'H', 'O12': 'X', 'O2C': 'N', 'O2G': 'G', 'OAD': 'N',
        'OAS': 'S', 'OBF': 'X', 'OBS': 'X', 'OCS': 'C', 'OCY': 'C',
        'ODP': 'N', 'OHI': 'H', 'OHS': 'D', 'OIC': 'X', 'OIP': 'I',
        'OLE': 'X', 'OLT': 'T', 'OLZ': 'S', 'OMC': 'C', 'OMG': 'G',
        'OMT': 'M', 'OMU': 'U', 'ONE': 'U', 'ONL': 'X', 'OPR': 'R',
        'ORN': 'A', 'ORQ': 'R', 'OSE': 'S', 'OTB': 'X', 'OTH': 'T',
        'OTY': 'Y', 'OXX': 'D', 'P  ': 'G', 'P1L': 'C', 'P1P': 'N',
        'P2T': 'T', 'P2U': 'U', 'P2Y': 'P', 'P5P': 'A', 'PAQ': 'Y',
        'PAS': 'D', 'PAT': 'W', 'PAU': 'A', 'PBB': 'C', 'PBF': 'F',
        'PBT': 'N', 'PCA': 'E', 'PCC': 'P', 'PCE': 'X', 'PCS': 'F',
        'PDL': 'X', 'PDU': 'U', 'PEC': 'C', 'PF5': 'F', 'PFF': 'F',
        'PFX': 'X', 'PG1': 'S', 'PG7': 'G', 'PG9': 'G', 'PGL': 'X',
        'PGN': 'G', 'PGP': 'G', 'PGY': 'G', 'PHA': 'F', 'PHD': 'D',
        'PHE': 'F', 'PHI': 'F', 'PHL': 'F', 'PHM': 'F', 'PIV': 'X',
        'PLE': 'L', 'PM3': 'F', 'PMT': 'C', 'POM': 'P', 'PPN': 'F',
        'PPU': 'A', 'PPW': 'G', 'PQ1': 'N', 'PR3': 'C', 'PR5': 'A',
        'PR9': 'P', 'PRN': 'A', 'PRO': 'P', 'PRS': 'P', 'PSA': 'F',
        'PSH': 'H', 'PST': 'T', 'PSU': 'U', 'PSW': 'C', 'PTA': 'X',
        'PTH': 'Y', 'PTM': 'Y', 'PTR': 'Y', 'PU ': 'A', 'PUY': 'N',
        'PVH': 'H', 'PVL': 'X', 'PYA': 'A', 'PYO': 'U', 'PYX': 'C',
        'PYY': 'N', 'QLG': 'QLG', 'QUO': 'G', 'R  ': 'A', 'R1A': 'C',
        'R1B': 'C', 'R1F': 'C', 'R7A': 'C', 'RC7': 'HYG', 'RCY': 'C',
        'RIA': 'A', 'RMP': 'A', 'RON': 'X', 'RT ': 'T', 'RTP': 'N',
        'S1H': 'S', 'S2C': 'C', 'S2D': 'A', 'S2M': 'T', 'S2P': 'A',
        'S4A': 'A', 'S4C': 'C', 'S4G': 'G', 'S4U': 'U', 'S6G': 'G',
        'SAC': 'S', 'SAH': 'C', 'SAR': 'G', 'SBL': 'S', 'SC ': 'C',
        'SCH': 'C', 'SCS': 'C', 'SCY': 'C', 'SD2': 'X', 'SDG': 'G',
        'SDP': 'S', 'SEB': 'S', 'SEC': 'A', 'SEG': 'A', 'SEL': 'S',
        'SEM': 'X', 'SEN': 'S', 'SEP': 'S', 'SER': 'S', 'SET': 'S',
        'SGB': 'S', 'SHC': 'C', 'SHP': 'G', 'SHR': 'K', 'SIB': 'C',
        'SIC': 'DC', 'SLA': 'P', 'SLR': 'P', 'SLZ': 'K', 'SMC': 'C',
        'SME': 'M', 'SMF': 'F', 'SMP': 'A', 'SMT': 'T', 'SNC': 'C',
        'SNN': 'N', 'SOC': 'C', 'SOS': 'N', 'SOY': 'S', 'SPT': 'T',
        'SRA': 'A', 'SSU': 'U', 'STY': 'Y', 'SUB': 'X', 'SUI': 'DG',
        'SUN': 'S', 'SUR': 'U', 'SVA': 'S', 'SVX': 'S', 'SVZ': 'X',
        'SYS': 'C', 'T  ': 'T', 'T11': 'F', 'T23': 'T', 'T2S': 'T',
        'T2T': 'N', 'T31': 'U', 'T32': 'T', 'T36': 'T', 'T37': 'T',
        'T38': 'T', 'T39': 'T', 'T3P': 'T', 'T41': 'T', 'T48': 'T',
        'T49': 'T', 'T4S': 'T', 'T5O': 'U', 'T5S': 'T', 'T66': 'X',
        'T6A': 'A', 'TA3': 'T', 'TA4': 'X', 'TAF': 'T', 'TAL': 'N',
        'TAV': 'D', 'TBG': 'V', 'TBM': 'T', 'TC1': 'C', 'TCP': 'T',
        'TCQ': 'X', 'TCR': 'W', 'TCY': 'A', 'TDD': 'L', 'TDY': 'T',
        'TFE': 'T', 'TFO': 'A', 'TFQ': 'F', 'TFT': 'T', 'TGP': 'G',
        'TH6': 'T', 'THC': 'T', 'THO': 'X', 'THR': 'T', 'THX': 'N',
        'THZ': 'R', 'TIH': 'A', 'TLB': 'N', 'TLC': 'T', 'TLN': 'U',
        'TMB': 'T', 'TMD': 'T', 'TNB': 'C', 'TNR': 'S', 'TOX': 'W',
        'TP1': 'T', 'TPC': 'C', 'TPG': 'G', 'TPH': 'X', 'TPL': 'W',
        'TPO': 'T', 'TPQ': 'Y', 'TQQ': 'W', 'TRF': 'W', 'TRG': 'K',
        'TRN': 'W', 'TRO': 'W', 'TRP': 'W', 'TRQ': 'W', 'TRW': 'W',
        'TRX': 'W', 'TS ': 'N', 'TST': 'X', 'TT ': 'N', 'TTD': 'T',
        'TTI': 'U', 'TTM': 'T', 'TTQ': 'W', 'TTS': 'Y', 'TY2': 'Y',
        'TY3': 'Y', 'TYB': 'Y', 'TYI': 'Y', 'TYN': 'Y', 'TYO': 'Y',
        'TYQ': 'Y', 'TYR': 'Y', 'TYS': 'Y', 'TYT': 'Y', 'TYU': 'N',
        'TYX': 'X', 'TYY': 'Y', 'TZB': 'X', 'TZO': 'X', 'U  ': 'U',
        'U25': 'U', 'U2L': 'U', 'U2N': 'U', 'U2P': 'U', 'U31': 'U',
        'U33': 'U', 'U34': 'U', 'U36': 'U', 'U37': 'U', 'U8U': 'U',
        'UAR': 'U', 'UCL': 'U', 'UD5': 'U', 'UDP': 'N', 'UFP': 'N',
        'UFR': 'U', 'UFT': 'U', 'UMA': 'A', 'UMP': 'U', 'UMS': 'U',
        'UN1': 'X', 'UN2': 'X', 'UNK': 'X', 'UR3': 'U', 'URD': 'U',
        'US1': 'U', 'US2': 'U', 'US3': 'T', 'US5': 'U', 'USM': 'U',
        'V1A': 'C', 'VAD': 'V', 'VAF': 'V', 'VAL': 'V', 'VB1': 'K',
        'VDL': 'X', 'VLL': 'X', 'VLM': 'X', 'VMS': 'X', 'VOL': 'X',
        'X  ': 'G', 'X2W': 'E', 'X4A': 'N', 'X9Q': 'AFG', 'XAD': 'A',
        'XAE': 'N', 'XAL': 'A', 'XAR': 'N', 'XCL': 'C', 'XCP': 'X',
        'XCR': 'C', 'XCS': 'N', 'XCT': 'C', 'XCY': 'C', 'XGA': 'N',
        'XGL': 'G', 'XGR': 'G', 'XGU': 'G', 'XTH': 'T', 'XTL': 'T',
        'XTR': 'T', 'XTS': 'G', 'XTY': 'N', 'XUA': 'A', 'XUG': 'G',
        'XX1': 'K', 'XXY': 'THG', 'XYG': 'DYG', 'Y  ': 'A', 'YCM': 'C',
        'YG ': 'G', 'YOF': 'Y', 'YRR': 'N', 'YYG': 'G', 'Z  ': 'C',
        'ZAD': 'A', 'ZAL': 'A', 'ZBC': 'C', 'ZCY': 'C', 'ZDU': 'U',
        'ZFB': 'X', 'ZGU': 'G', 'ZHP': 'N', 'ZTH': 'T', 'ZZJ': 'A'}

    # remove hetero atoms (waters/ligands/etc) from consideration?


    cmd.select("__h", "br. " + selName + " and not het")

    # get the AAs in the haystack
    aaDict = {'aaList': []}
    cmd.iterate("(name ca) and __h", "aaList.append((resi,resn,chain))", space=aaDict)

    IDs = [int(x[0]) for x in aaDict['aaList']]
    AAs = ''.join([one_letter[x[1]] for x in aaDict['aaList']])
    return AAs,IDs
def find_nucleotides_in_selection(selName=None, het=0, firstOnly=0):



    # remove hetero atoms (waters/ligands/etc) from consideration?


    cmd.select("__h", "br. " + selName + " and not het")

    # get the AAs in the haystack

    aaDict = {'aaList': [],'segments':[]}
    cmd.iterate(" __h", "aaList.append((segi,chain,resi,resn))", space=aaDict)

    # IDs = [int(x[0]) for x in aaDict['aaList']]
    # AAs = [x[1] for x in aaDict['aaDict']]
    return aaDict['aaList']

def findseq(needle, haystack, selName=None, het=0, firstOnly=0):
    # set the name of the selection to return.
    if selName == None:
        rSelName = "foundSeq" + str(random.randint(0, 32000))
        selName = rSelName
    elif selName == "sele":
        rSelName = "sele"
    else:
        rSelName = selName

    # input checking
    if not checkParams(needle, haystack, selName, het, firstOnly):
        print("There was an error with a parameter.  Please see")
        print("the above error message for how to fix it.")
        return None

    one_letter = {
        '00C': 'C', '01W': 'X', '0A0': 'D', '0A1': 'Y', '0A2': 'K',
        '0A8': 'C', '0AA': 'V', '0AB': 'V', '0AC': 'G', '0AD': 'G',
        '0AF': 'W', '0AG': 'L', '0AH': 'S', '0AK': 'D', '0AM': 'A',
        '0AP': 'C', '0AU': 'U', '0AV': 'A', '0AZ': 'P', '0BN': 'F',
        '0C ': 'C', '0CS': 'A', '0DC': 'C', '0DG': 'G', '0DT': 'T',
        '0G ': 'G', '0NC': 'A', '0SP': 'A', '0U ': 'U', '0YG': 'YG',
        '10C': 'C', '125': 'U', '126': 'U', '127': 'U', '128': 'N',
        '12A': 'A', '143': 'C', '175': 'ASG', '193': 'X', '1AP': 'A',
        '1MA': 'A', '1MG': 'G', '1PA': 'F', '1PI': 'A', '1PR': 'N',
        '1SC': 'C', '1TQ': 'W', '1TY': 'Y', '200': 'F', '23F': 'F',
        '23S': 'X', '26B': 'T', '2AD': 'X', '2AG': 'G', '2AO': 'X',
        '2AR': 'A', '2AS': 'X', '2AT': 'T', '2AU': 'U', '2BD': 'I',
        '2BT': 'T', '2BU': 'A', '2CO': 'C', '2DA': 'A', '2DF': 'N',
        '2DM': 'N', '2DO': 'X', '2DT': 'T', '2EG': 'G', '2FE': 'N',
        '2FI': 'N', '2FM': 'M', '2GT': 'T', '2HF': 'H', '2LU': 'L',
        '2MA': 'A', '2MG': 'G', '2ML': 'L', '2MR': 'R', '2MT': 'P',
        '2MU': 'U', '2NT': 'T', '2OM': 'U', '2OT': 'T', '2PI': 'X',
        '2PR': 'G', '2SA': 'N', '2SI': 'X', '2ST': 'T', '2TL': 'T',
        '2TY': 'Y', '2VA': 'V', '32S': 'X', '32T': 'X', '3AH': 'H',
        '3AR': 'X', '3CF': 'F', '3DA': 'A', '3DR': 'N', '3GA': 'A',
        '3MD': 'D', '3ME': 'U', '3NF': 'Y', '3TY': 'X', '3XH': 'G',
        '4AC': 'N', '4BF': 'Y', '4CF': 'F', '4CY': 'M', '4DP': 'W',
        '4F3': 'GYG', '4FB': 'P', '4FW': 'W', '4HT': 'W', '4IN': 'X',
        '4MF': 'N', '4MM': 'X', '4OC': 'C', '4PC': 'C', '4PD': 'C',
        '4PE': 'C', '4PH': 'F', '4SC': 'C', '4SU': 'U', '4TA': 'N',
        '5AA': 'A', '5AT': 'T', '5BU': 'U', '5CG': 'G', '5CM': 'C',
        '5CS': 'C', '5FA': 'A', '5FC': 'C', '5FU': 'U', '5HP': 'E',
        '5HT': 'T', '5HU': 'U', '5IC': 'C', '5IT': 'T', '5IU': 'U',
        '5MC': 'C', '5MD': 'N', '5MU': 'U', '5NC': 'C', '5PC': 'C',
        '5PY': 'T', '5SE': 'U', '5ZA': 'TWG', '64T': 'T', '6CL': 'K',
        '6CT': 'T', '6CW': 'W', '6HA': 'A', '6HC': 'C', '6HG': 'G',
        '6HN': 'K', '6HT': 'T', '6IA': 'A', '6MA': 'A', '6MC': 'A',
        '6MI': 'N', '6MT': 'A', '6MZ': 'N', '6OG': 'G', '70U': 'U',
        '7DA': 'A', '7GU': 'G', '7JA': 'I', '7MG': 'G', '8AN': 'A',
        '8FG': 'G', '8MG': 'G', '8OG': 'G', '9NE': 'E', '9NF': 'F',
        '9NR': 'R', '9NV': 'V', 'A  ': 'A', 'A1P': 'N', 'A23': 'A',
        'A2L': 'A', 'A2M': 'A', 'A34': 'A', 'A35': 'A', 'A38': 'A',
        'A39': 'A', 'A3A': 'A', 'A3P': 'A', 'A40': 'A', 'A43': 'A',
        'A44': 'A', 'A47': 'A', 'A5L': 'A', 'A5M': 'C', 'A5O': 'A',
        'A66': 'X', 'AA3': 'A', 'AA4': 'A', 'AAR': 'R', 'AB7': 'X',
        'ABA': 'A', 'ABR': 'A', 'ABS': 'A', 'ABT': 'N', 'ACB': 'D',
        'ACL': 'R', 'AD2': 'A', 'ADD': 'X', 'ADX': 'N', 'AEA': 'X',
        'AEI': 'D', 'AET': 'A', 'AFA': 'N', 'AFF': 'N', 'AFG': 'G',
        'AGM': 'R', 'AGT': 'X', 'AHB': 'N', 'AHH': 'X', 'AHO': 'A',
        'AHP': 'A', 'AHS': 'X', 'AHT': 'X', 'AIB': 'A', 'AKL': 'D',
        'ALA': 'A', 'ALC': 'A', 'ALG': 'R', 'ALM': 'A', 'ALN': 'A',
        'ALO': 'T', 'ALQ': 'X', 'ALS': 'A', 'ALT': 'A', 'ALY': 'K',
        'AP7': 'A', 'APE': 'X', 'APH': 'A', 'API': 'K', 'APK': 'K',
        'APM': 'X', 'APP': 'X', 'AR2': 'R', 'AR4': 'E', 'ARG': 'R',
        'ARM': 'R', 'ARO': 'R', 'ARV': 'X', 'AS ': 'A', 'AS2': 'D',
        'AS9': 'X', 'ASA': 'D', 'ASB': 'D', 'ASI': 'D', 'ASK': 'D',
        'ASL': 'D', 'ASM': 'X', 'ASN': 'N', 'ASP': 'D', 'ASQ': 'D',
        'ASU': 'N', 'ASX': 'B', 'ATD': 'T', 'ATL': 'T', 'ATM': 'T',
        'AVC': 'A', 'AVN': 'X', 'AYA': 'A', 'AYG': 'AYG', 'AZK': 'K',
        'AZS': 'S', 'AZY': 'Y', 'B1F': 'F', 'B1P': 'N', 'B2A': 'A',
        'B2F': 'F', 'B2I': 'I', 'B2V': 'V', 'B3A': 'A', 'B3D': 'D',
        'B3E': 'E', 'B3K': 'K', 'B3L': 'X', 'B3M': 'X', 'B3Q': 'X',
        'B3S': 'S', 'B3T': 'X', 'B3U': 'H', 'B3X': 'N', 'B3Y': 'Y',
        'BB6': 'C', 'BB7': 'C', 'BB9': 'C', 'BBC': 'C', 'BCS': 'C',
        'BCX': 'C', 'BE2': 'X', 'BFD': 'D', 'BG1': 'S', 'BGM': 'G',
        'BHD': 'D', 'BIF': 'F', 'BIL': 'X', 'BIU': 'I', 'BJH': 'X',
        'BLE': 'L', 'BLY': 'K', 'BMP': 'N', 'BMT': 'T', 'BNN': 'A',
        'BNO': 'X', 'BOE': 'T', 'BOR': 'R', 'BPE': 'C', 'BRU': 'U',
        'BSE': 'S', 'BT5': 'N', 'BTA': 'L', 'BTC': 'C', 'BTR': 'W',
        'BUC': 'C', 'BUG': 'V', 'BVP': 'U', 'BZG': 'N', 'C  ': 'C',
        'C12': 'TYG', 'C1X': 'K', 'C25': 'C', 'C2L': 'C', 'C2S': 'C',
        'C31': 'C', 'C32': 'C', 'C34': 'C', 'C36': 'C', 'C37': 'C',
        'C38': 'C', 'C3Y': 'C', 'C42': 'C', 'C43': 'C', 'C45': 'C',
        'C46': 'C', 'C49': 'C', 'C4R': 'C', 'C4S': 'C', 'C5C': 'C',
        'C66': 'X', 'C6C': 'C', 'C99': 'TFG', 'CAF': 'C', 'CAL': 'X',
        'CAR': 'C', 'CAS': 'C', 'CAV': 'X', 'CAY': 'C', 'CB2': 'C',
        'CBR': 'C', 'CBV': 'C', 'CCC': 'C', 'CCL': 'K', 'CCS': 'C',
        'CCY': 'CYG', 'CDE': 'X', 'CDV': 'X', 'CDW': 'C', 'CEA': 'C',
        'CFL': 'C', 'CFY': 'FCYG', 'CG1': 'G', 'CGA': 'E', 'CGU': 'E',
        'CH ': 'C', 'CH6': 'MYG', 'CH7': 'KYG', 'CHF': 'X', 'CHG': 'X',
        'CHP': 'G', 'CHS': 'X', 'CIR': 'R', 'CJO': 'GYG', 'CLE': 'L',
        'CLG': 'K', 'CLH': 'K', 'CLV': 'AFG', 'CM0': 'N', 'CME': 'C',
        'CMH': 'C', 'CML': 'C', 'CMR': 'C', 'CMT': 'C', 'CNU': 'U',
        'CP1': 'C', 'CPC': 'X', 'CPI': 'X', 'CQR': 'GYG', 'CR0': 'TLG',
        'CR2': 'GYG', 'CR5': 'G', 'CR7': 'KYG', 'CR8': 'HYG', 'CRF': 'TWG',
        'CRG': 'THG', 'CRK': 'MYG', 'CRO': 'GYG', 'CRQ': 'QYG', 'CRU': 'E',
        'CRW': 'ASG', 'CRX': 'ASG', 'CS0': 'C', 'CS1': 'C', 'CS3': 'C',
        'CS4': 'C', 'CS8': 'N', 'CSA': 'C', 'CSB': 'C', 'CSD': 'C',
        'CSE': 'C', 'CSF': 'C', 'CSH': 'SHG', 'CSI': 'G', 'CSJ': 'C',
        'CSL': 'C', 'CSO': 'C', 'CSP': 'C', 'CSR': 'C', 'CSS': 'C',
        'CSU': 'C', 'CSW': 'C', 'CSX': 'C', 'CSY': 'SYG', 'CSZ': 'C',
        'CTE': 'W', 'CTG': 'T', 'CTH': 'T', 'CUC': 'X', 'CWR': 'S',
        'CXM': 'M', 'CY0': 'C', 'CY1': 'C', 'CY3': 'C', 'CY4': 'C',
        'CYA': 'C', 'CYD': 'C', 'CYF': 'C', 'CYG': 'C', 'CYJ': 'X',
        'CYM': 'C', 'CYQ': 'C', 'CYR': 'C', 'CYS': 'C', 'CZ2': 'C',
        'CZO': 'GYG', 'CZZ': 'C', 'D11': 'T', 'D1P': 'N', 'D3 ': 'N',
        'D33': 'N', 'D3P': 'G', 'D3T': 'T', 'D4M': 'T', 'D4P': 'X',
        'DA ': 'A', 'DA2': 'X', 'DAB': 'A', 'DAH': 'F', 'DAL': 'A',
        'DAR': 'R', 'DAS': 'D', 'DBB': 'T', 'DBM': 'N', 'DBS': 'S',
        'DBU': 'T', 'DBY': 'Y', 'DBZ': 'A', 'DC ': 'C', 'DC2': 'C',
        'DCG': 'G', 'DCI': 'X', 'DCL': 'X', 'DCT': 'C', 'DCY': 'C',
        'DDE': 'H', 'DDG': 'G', 'DDN': 'U', 'DDX': 'N', 'DFC': 'C',
        'DFG': 'G', 'DFI': 'X', 'DFO': 'X', 'DFT': 'N', 'DG ': 'G',
        'DGH': 'G', 'DGI': 'G', 'DGL': 'E', 'DGN': 'Q', 'DHA': 'A',
        'DHI': 'H', 'DHL': 'X', 'DHN': 'V', 'DHP': 'X', 'DHU': 'U',
        'DHV': 'V', 'DI ': 'I', 'DIL': 'I', 'DIR': 'R', 'DIV': 'V',
        'DLE': 'L', 'DLS': 'K', 'DLY': 'K', 'DM0': 'K', 'DMH': 'N',
        'DMK': 'D', 'DMT': 'X', 'DN ': 'N', 'DNE': 'L', 'DNG': 'L',
        'DNL': 'K', 'DNM': 'L', 'DNP': 'A', 'DNR': 'C', 'DNS': 'K',
        'DOA': 'X', 'DOC': 'C', 'DOH': 'D', 'DON': 'L', 'DPB': 'T',
        'DPH': 'F', 'DPL': 'P', 'DPP': 'A', 'DPQ': 'Y', 'DPR': 'P',
        'DPY': 'N', 'DRM': 'U', 'DRP': 'N', 'DRT': 'T', 'DRZ': 'N',
        'DSE': 'S', 'DSG': 'N', 'DSN': 'S', 'DSP': 'D', 'DT ': 'T',
        'DTH': 'T', 'DTR': 'W', 'DTY': 'Y', 'DU ': 'U', 'DVA': 'V',
        'DXD': 'N', 'DXN': 'N', 'DYG': 'DYG', 'DYS': 'C', 'DZM': 'A',
        'E  ': 'A', 'E1X': 'A', 'EDA': 'A', 'EDC': 'G', 'EFC': 'C',
        'EHP': 'F', 'EIT': 'T', 'ENP': 'N', 'ESB': 'Y', 'ESC': 'M',
        'EXY': 'L', 'EY5': 'N', 'EYS': 'X', 'F2F': 'F', 'FA2': 'A',
        'FA5': 'N', 'FAG': 'N', 'FAI': 'N', 'FCL': 'F', 'FFD': 'N',
        'FGL': 'G', 'FGP': 'S', 'FHL': 'X', 'FHO': 'K', 'FHU': 'U',
        'FLA': 'A', 'FLE': 'L', 'FLT': 'Y', 'FME': 'M', 'FMG': 'G',
        'FMU': 'N', 'FOE': 'C', 'FOX': 'G', 'FP9': 'P', 'FPA': 'F',
        'FRD': 'X', 'FT6': 'W', 'FTR': 'W', 'FTY': 'Y', 'FZN': 'K',
        'G  ': 'G', 'G25': 'G', 'G2L': 'G', 'G2S': 'G', 'G31': 'G',
        'G32': 'G', 'G33': 'G', 'G36': 'G', 'G38': 'G', 'G42': 'G',
        'G46': 'G', 'G47': 'G', 'G48': 'G', 'G49': 'G', 'G4P': 'N',
        'G7M': 'G', 'GAO': 'G', 'GAU': 'E', 'GCK': 'C', 'GCM': 'X',
        'GDP': 'G', 'GDR': 'G', 'GFL': 'G', 'GGL': 'E', 'GH3': 'G',
        'GHG': 'Q', 'GHP': 'G', 'GL3': 'G', 'GLH': 'Q', 'GLM': 'X',
        'GLN': 'Q', 'GLQ': 'E', 'GLU': 'E', 'GLX': 'Z', 'GLY': 'G',
        'GLZ': 'G', 'GMA': 'E', 'GMS': 'G', 'GMU': 'U', 'GN7': 'G',
        'GND': 'X', 'GNE': 'N', 'GOM': 'G', 'GPL': 'K', 'GS ': 'G',
        'GSC': 'G', 'GSR': 'G', 'GSS': 'G', 'GSU': 'E', 'GT9': 'C',
        'GTP': 'G', 'GVL': 'X', 'GYC': 'CYG', 'GYS': 'SYG', 'H2U': 'U',
        'H5M': 'P', 'HAC': 'A', 'HAR': 'R', 'HBN': 'H', 'HCS': 'X',
        'HDP': 'U', 'HEU': 'U', 'HFA': 'X', 'HGL': 'X', 'HHI': 'H',
        'HHK': 'AK', 'HIA': 'H', 'HIC': 'H', 'HIP': 'H', 'HIQ': 'H',
        'HIS': 'H', 'HL2': 'L', 'HLU': 'L', 'HMF': 'A', 'HMR': 'R',
        'HOL': 'N', 'HPC': 'F', 'HPE': 'F', 'HPQ': 'F', 'HQA': 'A',
        'HRG': 'R', 'HRP': 'W', 'HS8': 'H', 'HS9': 'H', 'HSE': 'S',
        'HSL': 'S', 'HSO': 'H', 'HTI': 'C', 'HTN': 'N', 'HTR': 'W',
        'HV5': 'A', 'HVA': 'V', 'HY3': 'P', 'HYP': 'P', 'HZP': 'P',
        'I  ': 'I', 'I2M': 'I', 'I58': 'K', 'I5C': 'C', 'IAM': 'A',
        'IAR': 'R', 'IAS': 'D', 'IC ': 'C', 'IEL': 'K', 'IEY': 'HYG',
        'IG ': 'G', 'IGL': 'G', 'IGU': 'G', 'IIC': 'SHG', 'IIL': 'I',
        'ILE': 'I', 'ILG': 'E', 'ILX': 'I', 'IMC': 'C', 'IML': 'I',
        'IOY': 'F', 'IPG': 'G', 'IPN': 'N', 'IRN': 'N', 'IT1': 'K',
        'IU ': 'U', 'IYR': 'Y', 'IYT': 'T', 'JJJ': 'C', 'JJK': 'C',
        'JJL': 'C', 'JW5': 'N', 'K1R': 'C', 'KAG': 'G', 'KCX': 'K',
        'KGC': 'K', 'KOR': 'M', 'KPI': 'K', 'KST': 'K', 'KYQ': 'K',
        'L2A': 'X', 'LA2': 'K', 'LAA': 'D', 'LAL': 'A', 'LBY': 'K',
        'LC ': 'C', 'LCA': 'A', 'LCC': 'N', 'LCG': 'G', 'LCH': 'N',
        'LCK': 'K', 'LCX': 'K', 'LDH': 'K', 'LED': 'L', 'LEF': 'L',
        'LEH': 'L', 'LEI': 'V', 'LEM': 'L', 'LEN': 'L', 'LET': 'X',
        'LEU': 'L', 'LG ': 'G', 'LGP': 'G', 'LHC': 'X', 'LHU': 'U',
        'LKC': 'N', 'LLP': 'K', 'LLY': 'K', 'LME': 'E', 'LMQ': 'Q',
        'LMS': 'N', 'LP6': 'K', 'LPD': 'P', 'LPG': 'G', 'LPL': 'X',
        'LPS': 'S', 'LSO': 'X', 'LTA': 'X', 'LTR': 'W', 'LVG': 'G',
        'LVN': 'V', 'LYM': 'K', 'LYN': 'K', 'LYR': 'K', 'LYS': 'K',
        'LYX': 'K', 'LYZ': 'K', 'M0H': 'C', 'M1G': 'G', 'M2G': 'G',
        'M2L': 'K', 'M2S': 'M', 'M3L': 'K', 'M5M': 'C', 'MA ': 'A',
        'MA6': 'A', 'MA7': 'A', 'MAA': 'A', 'MAD': 'A', 'MAI': 'R',
        'MBQ': 'Y', 'MBZ': 'N', 'MC1': 'S', 'MCG': 'X', 'MCL': 'K',
        'MCS': 'C', 'MCY': 'C', 'MDH': 'X', 'MDO': 'ASG', 'MDR': 'N',
        'MEA': 'F', 'MED': 'M', 'MEG': 'E', 'MEN': 'N', 'MEP': 'U',
        'MEQ': 'Q', 'MET': 'M', 'MEU': 'G', 'MF3': 'X', 'MFC': 'GYG',
        'MG1': 'G', 'MGG': 'R', 'MGN': 'Q', 'MGQ': 'A', 'MGV': 'G',
        'MGY': 'G', 'MHL': 'L', 'MHO': 'M', 'MHS': 'H', 'MIA': 'A',
        'MIS': 'S', 'MK8': 'L', 'ML3': 'K', 'MLE': 'L', 'MLL': 'L',
        'MLY': 'K', 'MLZ': 'K', 'MME': 'M', 'MMT': 'T', 'MND': 'N',
        'MNL': 'L', 'MNU': 'U', 'MNV': 'V', 'MOD': 'X', 'MP8': 'P',
        'MPH': 'X', 'MPJ': 'X', 'MPQ': 'G', 'MRG': 'G', 'MSA': 'G',
        'MSE': 'M', 'MSL': 'M', 'MSO': 'M', 'MSP': 'X', 'MT2': 'M',
        'MTR': 'T', 'MTU': 'A', 'MTY': 'Y', 'MVA': 'V', 'N  ': 'N',
        'N10': 'S', 'N2C': 'X', 'N5I': 'N', 'N5M': 'C', 'N6G': 'G',
        'N7P': 'P', 'NA8': 'A', 'NAL': 'A', 'NAM': 'A', 'NB8': 'N',
        'NBQ': 'Y', 'NC1': 'S', 'NCB': 'A', 'NCX': 'N', 'NCY': 'X',
        'NDF': 'F', 'NDN': 'U', 'NEM': 'H', 'NEP': 'H', 'NF2': 'N',
        'NFA': 'F', 'NHL': 'E', 'NIT': 'X', 'NIY': 'Y', 'NLE': 'L',
        'NLN': 'L', 'NLO': 'L', 'NLP': 'L', 'NLQ': 'Q', 'NMC': 'G',
        'NMM': 'R', 'NMS': 'T', 'NMT': 'T', 'NNH': 'R', 'NP3': 'N',
        'NPH': 'C', 'NRP': 'LYG', 'NRQ': 'MYG', 'NSK': 'X', 'NTY': 'Y',
        'NVA': 'V', 'NYC': 'TWG', 'NYG': 'NYG', 'NYM': 'N', 'NYS': 'C',
        'NZH': 'H', 'O12': 'X', 'O2C': 'N', 'O2G': 'G', 'OAD': 'N',
        'OAS': 'S', 'OBF': 'X', 'OBS': 'X', 'OCS': 'C', 'OCY': 'C',
        'ODP': 'N', 'OHI': 'H', 'OHS': 'D', 'OIC': 'X', 'OIP': 'I',
        'OLE': 'X', 'OLT': 'T', 'OLZ': 'S', 'OMC': 'C', 'OMG': 'G',
        'OMT': 'M', 'OMU': 'U', 'ONE': 'U', 'ONL': 'X', 'OPR': 'R',
        'ORN': 'A', 'ORQ': 'R', 'OSE': 'S', 'OTB': 'X', 'OTH': 'T',
        'OTY': 'Y', 'OXX': 'D', 'P  ': 'G', 'P1L': 'C', 'P1P': 'N',
        'P2T': 'T', 'P2U': 'U', 'P2Y': 'P', 'P5P': 'A', 'PAQ': 'Y',
        'PAS': 'D', 'PAT': 'W', 'PAU': 'A', 'PBB': 'C', 'PBF': 'F',
        'PBT': 'N', 'PCA': 'E', 'PCC': 'P', 'PCE': 'X', 'PCS': 'F',
        'PDL': 'X', 'PDU': 'U', 'PEC': 'C', 'PF5': 'F', 'PFF': 'F',
        'PFX': 'X', 'PG1': 'S', 'PG7': 'G', 'PG9': 'G', 'PGL': 'X',
        'PGN': 'G', 'PGP': 'G', 'PGY': 'G', 'PHA': 'F', 'PHD': 'D',
        'PHE': 'F', 'PHI': 'F', 'PHL': 'F', 'PHM': 'F', 'PIV': 'X',
        'PLE': 'L', 'PM3': 'F', 'PMT': 'C', 'POM': 'P', 'PPN': 'F',
        'PPU': 'A', 'PPW': 'G', 'PQ1': 'N', 'PR3': 'C', 'PR5': 'A',
        'PR9': 'P', 'PRN': 'A', 'PRO': 'P', 'PRS': 'P', 'PSA': 'F',
        'PSH': 'H', 'PST': 'T', 'PSU': 'U', 'PSW': 'C', 'PTA': 'X',
        'PTH': 'Y', 'PTM': 'Y', 'PTR': 'Y', 'PU ': 'A', 'PUY': 'N',
        'PVH': 'H', 'PVL': 'X', 'PYA': 'A', 'PYO': 'U', 'PYX': 'C',
        'PYY': 'N', 'QLG': 'QLG', 'QUO': 'G', 'R  ': 'A', 'R1A': 'C',
        'R1B': 'C', 'R1F': 'C', 'R7A': 'C', 'RC7': 'HYG', 'RCY': 'C',
        'RIA': 'A', 'RMP': 'A', 'RON': 'X', 'RT ': 'T', 'RTP': 'N',
        'S1H': 'S', 'S2C': 'C', 'S2D': 'A', 'S2M': 'T', 'S2P': 'A',
        'S4A': 'A', 'S4C': 'C', 'S4G': 'G', 'S4U': 'U', 'S6G': 'G',
        'SAC': 'S', 'SAH': 'C', 'SAR': 'G', 'SBL': 'S', 'SC ': 'C',
        'SCH': 'C', 'SCS': 'C', 'SCY': 'C', 'SD2': 'X', 'SDG': 'G',
        'SDP': 'S', 'SEB': 'S', 'SEC': 'A', 'SEG': 'A', 'SEL': 'S',
        'SEM': 'X', 'SEN': 'S', 'SEP': 'S', 'SER': 'S', 'SET': 'S',
        'SGB': 'S', 'SHC': 'C', 'SHP': 'G', 'SHR': 'K', 'SIB': 'C',
        'SIC': 'DC', 'SLA': 'P', 'SLR': 'P', 'SLZ': 'K', 'SMC': 'C',
        'SME': 'M', 'SMF': 'F', 'SMP': 'A', 'SMT': 'T', 'SNC': 'C',
        'SNN': 'N', 'SOC': 'C', 'SOS': 'N', 'SOY': 'S', 'SPT': 'T',
        'SRA': 'A', 'SSU': 'U', 'STY': 'Y', 'SUB': 'X', 'SUI': 'DG',
        'SUN': 'S', 'SUR': 'U', 'SVA': 'S', 'SVX': 'S', 'SVZ': 'X',
        'SYS': 'C', 'T  ': 'T', 'T11': 'F', 'T23': 'T', 'T2S': 'T',
        'T2T': 'N', 'T31': 'U', 'T32': 'T', 'T36': 'T', 'T37': 'T',
        'T38': 'T', 'T39': 'T', 'T3P': 'T', 'T41': 'T', 'T48': 'T',
        'T49': 'T', 'T4S': 'T', 'T5O': 'U', 'T5S': 'T', 'T66': 'X',
        'T6A': 'A', 'TA3': 'T', 'TA4': 'X', 'TAF': 'T', 'TAL': 'N',
        'TAV': 'D', 'TBG': 'V', 'TBM': 'T', 'TC1': 'C', 'TCP': 'T',
        'TCQ': 'X', 'TCR': 'W', 'TCY': 'A', 'TDD': 'L', 'TDY': 'T',
        'TFE': 'T', 'TFO': 'A', 'TFQ': 'F', 'TFT': 'T', 'TGP': 'G',
        'TH6': 'T', 'THC': 'T', 'THO': 'X', 'THR': 'T', 'THX': 'N',
        'THZ': 'R', 'TIH': 'A', 'TLB': 'N', 'TLC': 'T', 'TLN': 'U',
        'TMB': 'T', 'TMD': 'T', 'TNB': 'C', 'TNR': 'S', 'TOX': 'W',
        'TP1': 'T', 'TPC': 'C', 'TPG': 'G', 'TPH': 'X', 'TPL': 'W',
        'TPO': 'T', 'TPQ': 'Y', 'TQQ': 'W', 'TRF': 'W', 'TRG': 'K',
        'TRN': 'W', 'TRO': 'W', 'TRP': 'W', 'TRQ': 'W', 'TRW': 'W',
        'TRX': 'W', 'TS ': 'N', 'TST': 'X', 'TT ': 'N', 'TTD': 'T',
        'TTI': 'U', 'TTM': 'T', 'TTQ': 'W', 'TTS': 'Y', 'TY2': 'Y',
        'TY3': 'Y', 'TYB': 'Y', 'TYI': 'Y', 'TYN': 'Y', 'TYO': 'Y',
        'TYQ': 'Y', 'TYR': 'Y', 'TYS': 'Y', 'TYT': 'Y', 'TYU': 'N',
        'TYX': 'X', 'TYY': 'Y', 'TZB': 'X', 'TZO': 'X', 'U  ': 'U',
        'U25': 'U', 'U2L': 'U', 'U2N': 'U', 'U2P': 'U', 'U31': 'U',
        'U33': 'U', 'U34': 'U', 'U36': 'U', 'U37': 'U', 'U8U': 'U',
        'UAR': 'U', 'UCL': 'U', 'UD5': 'U', 'UDP': 'N', 'UFP': 'N',
        'UFR': 'U', 'UFT': 'U', 'UMA': 'A', 'UMP': 'U', 'UMS': 'U',
        'UN1': 'X', 'UN2': 'X', 'UNK': 'X', 'UR3': 'U', 'URD': 'U',
        'US1': 'U', 'US2': 'U', 'US3': 'T', 'US5': 'U', 'USM': 'U',
        'V1A': 'C', 'VAD': 'V', 'VAF': 'V', 'VAL': 'V', 'VB1': 'K',
        'VDL': 'X', 'VLL': 'X', 'VLM': 'X', 'VMS': 'X', 'VOL': 'X',
        'X  ': 'G', 'X2W': 'E', 'X4A': 'N', 'X9Q': 'AFG', 'XAD': 'A',
        'XAE': 'N', 'XAL': 'A', 'XAR': 'N', 'XCL': 'C', 'XCP': 'X',
        'XCR': 'C', 'XCS': 'N', 'XCT': 'C', 'XCY': 'C', 'XGA': 'N',
        'XGL': 'G', 'XGR': 'G', 'XGU': 'G', 'XTH': 'T', 'XTL': 'T',
        'XTR': 'T', 'XTS': 'G', 'XTY': 'N', 'XUA': 'A', 'XUG': 'G',
        'XX1': 'K', 'XXY': 'THG', 'XYG': 'DYG', 'Y  ': 'A', 'YCM': 'C',
        'YG ': 'G', 'YOF': 'Y', 'YRR': 'N', 'YYG': 'G', 'Z  ': 'C',
        'ZAD': 'A', 'ZAL': 'A', 'ZBC': 'C', 'ZCY': 'C', 'ZDU': 'U',
        'ZFB': 'X', 'ZGU': 'G', 'ZHP': 'N', 'ZTH': 'T', 'ZZJ': 'A'}

    # remove hetero atoms (waters/ligands/etc) from consideration?
    if het:
        cmd.select("__h", "br. " + haystack)
    else:
        cmd.select("__h", "br. " + haystack + " and not het")

    # get the AAs in the haystack
    aaDict = {'aaList': []}
    cmd.iterate("(name ca) and __h", "aaList.append((resi,resn,chain))", space=aaDict)

    IDs = [int(x[0]) for x in aaDict['aaList']]
    AAs = ''.join([one_letter[x[1]] for x in aaDict['aaList']])
    chains = [x[2] for x in aaDict['aaList']]

    reNeedle = re.compile(needle.upper())
    it = reNeedle.finditer(AAs)

    # make an empty selection to which we add residues
    cmd.select(rSelName, 'None')

    for i in it:
        (start, stop) = i.span()
        # we found some residues, which chains are they from?
        i_chains = chains[start:stop]
        # are all residues from one chain?
        if len(set(i_chains)) != 1:
        	# now they are not, this match is not really a match, skip it
        	continue
        chain = i_chains[0]
        cmd.select(rSelName, rSelName + " or (__h and i. " + str(IDs[start]) + "-" + str(IDs[stop - 1]) + " and c. " + chain + " )")
        if int(firstOnly):
            break
    cmd.delete("__h")
    return rSelName
cmd.extend("findseq", findseq)


def checkParams(needle, haystack, selName, het, firstOnly):
    """
    This is just a helper function for checking the user input
    """
    # check Needle
    if len(needle) == 0 or not cmd.is_string(needle):
        print("Error: Please provide a string 'needle' to search for.")
        print("Error: For help type 'help motifFinder'.")
        return False

    # check Haystack
    if len(haystack) == 0 or not cmd.is_string(haystack):
        print("Error: Please provide valid PyMOL object or selection name")
        print("Error: in which to search.")
        print("Error: For help type 'help motifFinder'.")
        return False

    # check het
    try:
        het = bool(int(het))
    except ValueError:
        print("Error: The 'het' parameter was not 0 or 1.")
        return False

    # check first Only
    try:
        firstOnly = bool(int(het))
    except ValueError:
        print("Error: The 'firstOnly' parameter was not 0 or 1.")
        return False

    # check selName
    if not cmd.is_string(selName):
        print("Error: selName was not a string.")
        return False
    return True

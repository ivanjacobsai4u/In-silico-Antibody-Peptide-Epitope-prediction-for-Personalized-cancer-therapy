# from pymnet import MultilayerNetwork
# from pymnet import draw as pymnet_draw
# def vizualize_layers(A,show=True,layerOrderDict={},figsize=(42, 25),nodeSizeRule={"rule": "degree", "propscale": 0.0005}):
#     '''
#
#     Args:
#         A: multiplex
#         show: show plot
#         layerOrderDict: the order of the layers for vizualization
#         figsize: the size of the figure
#         nodeSizeRule: the rule for the node size e.g. degree
#
#     Returns:
#
#     '''
#
#     mplex = MultilayerNetwork(aspects=1);
#     for g in A.get_edges(True):
#              mplex[g[0][0], g[1][0], g[0][1], g[1][1]] = 1
#
#     pymnet_draw(mplex, show=show,
#                                       figsize=figsize, layerPadding=0.6, layergap=0.6, defaultLayerAlpha=0.3,
#                                       layout="spring", elev=8, azim=5,
#                                       nodeColorDict={(0, 0): "r", (1, 0): "r", (0, 1): "r"},
#                                       nodeLabelRule={}, defaultLayerLabelLoc=(0, 1),
#                                       layerOrderDict=layerOrderDict,
#                                       defaultLayerLabelSize=18,
#                                       edgeColorRule={"rule": "edgeweight", "colormap": "jet", "scaleby": 0.1},
#                                       nodeSizeRule=nodeSizeRule);


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
import pandas as pd
import csv
from Bio import SeqIO
def filter_cedar(tcell_table_file,
                 hlalleles_prot_fastas_seq,
                 allellist_file,IEDB_response_code={'Positive': 1,
                                     'Positive-High': 1,
                                     'Positive-Intermediate': 1,
                                     'Positive-Low': 1,
                                     'Negative': 0,
                                     },cell_tissue_type=None):
        allellist = pd.read_csv(allellist_file, sep=",")
        allellist = allellist.astype({column: 'string' for column in allellist.columns})
        fasta_sequences = SeqIO.parse(open(hlalleles_prot_fastas_seq), 'fasta')
        alles_fastea_seq_df = pd.DataFrame(
                [{'AlleleID': str(fasta.id), "Description": fasta.description, "Sequence": str(fasta.seq)} for fasta in
                 fasta_sequences])
        alles_fastea_seq_df = alles_fastea_seq_df.astype(
                {column: 'string' for column in alles_fastea_seq_df.columns})
        alles_fastea_seq_df['AlleleID'] = alles_fastea_seq_df['AlleleID'].apply(lambda x: x.replace('HLA:', ''))

        allellist = allellist.merge(alles_fastea_seq_df, on='AlleleID')
        allellist = allellist.drop_duplicates()
        allellist['Sequence'] = allellist['Sequence'].str.strip()

        with open(tcell_table_file, 'r', encoding='utf-8') as input_handle:
                csv_reader = csv.reader(input_handle, delimiter=',')
                header_1 = next(csv_reader)
                header_2 = next(csv_reader)
                header_list = []
                for x, y in zip(header_1, header_2):
                        header_list.append(':'.join([x, y]))

        mhc_full = pd.read_csv(tcell_table_file, skiprows=2)
        mhc_full.columns = header_list
        mhc_full['Allele'] = mhc_full['MHC:Allele Name'].apply(lambda x: str(x).replace("HLA-", ''))
        mhc_full['Epitope:Description'] = mhc_full['Epitope:Description'].str.strip()
        idx = mhc_full['Epitope:Description'].str.contains(' ')
        mhc_full.loc[idx, 'Epitope:Description'] = mhc_full.loc[idx, 'Epitope:Description'].apply(
                lambda x: x[0:x.index(' ')])

        mhc_full['AlleleOrig'] = mhc_full.Allele
        allellist['AlleleOrig'] = allellist.Allele
        mhc_full['Allele'] = mhc_full.Allele.apply(
                lambda x: ''.join(x.split(':')[0:1]) + ':' + ''.join(x.split(':')[1:2]))
        allellist['Allele'] = allellist.Allele.apply(
                lambda x: ''.join(x.split(':')[0:1]) + ':' + ''.join(x.split(':')[1:2]))

        mhc_filtered = mhc_full[(mhc_full['Allele'].isin(allellist.Allele))]

        mhc_filtered['Sequence'] = mhc_filtered.Allele.apply(
                lambda x: allellist.loc[allellist.Allele == x, 'Sequence'].unique()[0])

        mhc_filtered.rename(columns={"Description": "Allele:Description", "Sequence": "Allele:Sequence"})
        mhc_filtered['y'] = mhc_filtered['Assay:Qualitative Measure'].apply(lambda x: IEDB_response_code[x])

        filter = (mhc_filtered['Epitope:Parent Species'] == 'Homo sapiens') & (
                mhc_filtered['Epitope:Description'].str.len() <= 25) & (
                         mhc_filtered['Epitope:Description'].str.len() >= 8) & (
                         mhc_filtered['MHC:Class'] == 'I')
        if cell_tissue_type is not None:
                filter = (mhc_filtered['Epitope:Parent Species'] == 'Homo sapiens') & (
                        mhc_filtered['Epitope:Description'].str.len() <= 25) & (
                                 mhc_filtered['Epitope:Description'].str.len() >= 8) & (
                                 mhc_filtered['MHC:Class'] == 'I') & (
                                 mhc_filtered['Antigen Presenting Cells:Cell Tissue Type'] == cell_tissue_type)
        mhc_filtered = mhc_filtered[filter]
        return mhc_filtered
#
# output_file = "training_data_copy.csv"
# tcell_table_file = 'data/tcell_full_v3.csv'
# hlalleles_prot_fastas_seq = 'data/hla_prot.fasta'
# allellist_file = 'data/Allelelist.txt'
# mhc_filtered=filter_cedar(tcell_table_file,
#                  hlalleles_prot_fastas_seq,
#                  allellist_file)
#

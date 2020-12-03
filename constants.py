# REFERENCES PARAMETERS
CONFIG = {
    "epochs": 200,
    "batch_size": 32,
    "num_classes": 5,
    # "num_models": 1,
    "num_models": 36,
    "dataset": "kdd",
    "train_data": "train+",
    "img_rows": 11,
    "img_cols": 11,
    "output_dim": 121,
    "num_process": 6,
    "smote_rate": 1,
    "model_type": "cnn",
    "support_rate": 10,
}

LABEL_TO_NUM = {
    "normal": 0,
    "probe": 1,
    "dos": 2,
    "u2r": 3,
    "r2l": 4,
}

# NSL-KDD dataset
# Names of the 41 features
FULL_FEATURES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]

# Names of all the attacks names (including NSL KDD)
ENTRY_TYPE = {
    "normal": [
        "normal.",
    ],
    "probe": [
        "ipsweep.",
        "nmap.",
        "portsweep.",
        "satan.",
        "saint.",
        "mscan.",
    ],
    "dos": [
        "back.",
        "land.",
        "neptune.",
        "pod.",
        "smurf.",
        "teardrop.",
        "apache2.",
        "udpstorm.",
        "processtable.",
        "mailbomb.",
    ],
    "u2r": [
        "buffer_overflow.",
        "loadmodule.",
        "perl.",
        "rootkit.",
        "xterm.",
        "ps.",
        "sqlattack.",
        "httptunnel.",  # こっち？
    ],
    "r2l": [
        "ftp_write.",
        "guess_passwd.",
        "imap.",
        "multihop.",
        "phf.",
        "spy.",
        "warezclient.",
        "warezmaster.",
        "snmpgetattack.",
        "named.",
        "xlock.",
        "xsnoop.",
        "sendmail.",
        # "httptunnel.",
        "worm.",
        "snmpguess.",
    ]
}

__TRAIN_SAMLE_NUM_PER_LABEL = [
    96,
    71,
    63,
    69,
    71,
    0,
    0,
    59,
    25,
    92,
    46,
    68,
    59,
    0,
    0,
    0,
    0,
    30,
    20,
    12,
    21,
    0,
    0,
    0,
    0,
    19,
    34,
    21,
    18,
    14,
    9,
    59,
    26,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]

SAMPLE_NUM_PER_LABEL = {
    # normal
    "normal.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[0],
        "test": 10 * CONFIG["smote_rate"]
    },
    # probe
    "ipsweep.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[1],
        "test":  2,
    },
    "nmap.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[2],
        "test": 0,
    },
    "portsweep.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[3],
        "test":  2,
    },
    "satan.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[4],
        "test":  2,
    },
    "saint.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[5],
        "test":  2,
    },
    "mscan.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[6],
        "test":  2,
    },
    # dos
    "back.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[7],
        "test": 2,
    },
    "land.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[8],
        "test": 0,
    },
    "neptune.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[9],
        "test": 2,
    },
    "pod.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[10],
        "test": 0,
    },
    "smurf.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[11],
        "test": 2,
    },
    "teardrop.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[12],
        "test": 0,
    },
    "apache2.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[13],
        "test": 2,
    },
    "udpstorm.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[14],
        "test": 0,
    },
    "processtable.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[15],
        "test": 2,
    },
    "mailbomb.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[16],
        "test": 0,
    },
    # u2r
    "buffer_overflow.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[17],
        "test": 2,
    },
    "loadmodule.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[18],
        "test": 0,
    },
    "perl.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[19],
        "test": 0,
    },
    "rootkit.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[20],
        "test": 2,
    },
    "xterm.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[21],
        "test": 2,
    },
    "ps.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[22],
        "test": 2,
    },
    "sqlattack.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[23],
        "test": 0,
    },
    "httptunnel.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[24],
        "test": 2,
    },
    # r2l
    "ftp_write.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[25],
        "test": 0,
    },
    "guess_passwd.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[26],
        "test": 2,
    },
    "imap.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[27],
        "test": 0,
    },
    "multihop.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[28],
        "test": 2,
    },
    "phf.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[29],
        "test": 0,
    },
    "spy.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[30],
        "test": 0,
    },
    "warezclient.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[31],
        "test": 0,
    },
    "warezmaster.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[32],
        "test": 2,
    },
    "snmpgetattack.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[33],
        "test": 2,
    },
    "named.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[34],
        "test": 0,
    },
    "xlock.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[35],
        "test": 0,
    },
    "xsnoop.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[36],
        "test": 0,
    },
    "sendmail.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[37],
        "test": 0,
    },
    "worm.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[38],
        "test": 0,
    },
    "snmpguess.": {
        "train": __TRAIN_SAMLE_NUM_PER_LABEL[39],
        "test": 2,
    },
}

# ***** KDD STRING FEATURES VALUES *****
SERVICE_VALUES = [
    "http",
    "smtp",
    "finger",
    "domain_u",
    "auth",
    "telnet",
    "ftp",
    "eco_i",
    "ntp_u",
    "ecr_i",
    "other",
    "private",
    "pop_3",
    "ftp_data",
    "rje",
    "time",
    "mtp",
    "link",
    "remote_job",
    "gopher",
    "ssh",
    "name",
    "whois",
    "domain",
    "login",
    "imap4",
    "daytime",
    "ctf",
    "nntp",
    "shell",
    "IRC",
    "nnsp",
    "http_443",
    "exec",
    "printer",
    "efs",
    "courier",
    "uucp",
    "klogin",
    "kshell",
    "echo",
    "discard",
    "systat",
    "supdup",
    "iso_tsap",
    "hostnames",
    "csnet_ns",
    "pop_2",
    "sunrpc",
    "uucp_path",
    "netbios_ns",
    "netbios_ssn",
    "netbios_dgm",
    "sql_net",
    "vmnet",
    "bgp",
    "Z39_50",
    "ldap",
    "netstat",
    "urh_i",
    "X11",
    "urp_i",
    "pm_dump",
    "tftp_u",
    "tim_i",
    "red_i",
    "icmp",
    "http_2784",
    "harvest",
    "aol",
    "http_8001",
]

FLAG_VALUES = [
    "OTH",
    "RSTOS0",
    "SF",
    "SH",
    "RSTO",
    "S2",
    "S1",
    "REJ",
    "S3",
    "RSTR",
    "S0",
]

PROTOCOL_TYPE_VALUES = [
    "tcp",
    "udp",
    "icmp",
]

_COLUMNS = [
    "label",
    "src_bytes", "dst_bytes", "duration", "logged_in", "dst_host_count", "dst_host_srv_count", "serror_rate", "tcp", "udp", "SH", "REJ",
    "land", "wrong_fragment", "num_failed_logins", "num_compromised", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "srv_serror_rate", "same_srv_rate", "OTH", "RSTO", "S3",
    "urgent", "hot", "root_shell", "su_attempted", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "rerror_rate", "diff_srv_rate", "RSTOS0", "S2", "RSTR",
    "count", "srv_count", "num_root", "num_file_creations", "dst_host_serror_rate", "dst_host_srv_serror_rate", "srv_rerror_rate", "srv_diff_host_rate", "SF", "S1", "S0",
    "is_host_login", "is_guest_login", "num_shells", "num_access_files", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "http", "smtp", "finger", "domain_u", "auth",
    "telnet", "ftp", "eco_i", "ntp_u", "ecr_i", "other", "private", "pop_3", "ftp_data", "rje", "time",
    "mtp", "link", "remote_job", "gopher", "ssh", "name", "whois", "domain", "login", "imap4", "daytime",
    "ctf", "nntp", "shell", "IRC", "nnsp", "http_443", "exec", "printer", "efs", "courier", "uucp",
    "klogin", "kshell", "echo", "discard", "systat", "supdup", "iso_tsap", "hostnames", "csnet_ns", "pop_2", "sunrpc",
    "uucp_path", "netbios_ns", "netbios_ssn", "netbios_dgm", "sql_net", "vmnet", "bgp", "Z39_50", "ldap", "netstat", "urh_i",
    "X11", "urp_i", "pm_dump", "tftp_u", "tim_i", "red_i", "icmp", "http_2784", "harvest", "aol", "http_8001",
]

BASE_COLUMNS = [
    "label",  # -1
    "duration",  # 0
    "tcp",  # 1 PROTOCOL
    "udp",  # 2 PROTOCOL
    "http",  # 3 SERVICE
    "smtp",  # 4 SERVICE
    "finger",  # 5 SERVICE
    "domain_u",  # 6 SERVICE
    "auth",  # 7 SERVICE
    "telnet",  # 8 SERVICE
    "ftp",  # 9 SERVICE
    "eco_i",  # 10 SERVICE
    "ntp_u",  # 11 SERVICE
    "ecr_i",  # 12 SERVICE
    "other",  # 13 SERVICE
    "private",  # 14 SERVICE
    "pop_3",  # 15 SERVICE
    "ftp_data",  # 16 SERVICE
    "rje",  # 17 SERVICE
    "time",  # 18 SERVICE
    "mtp",  # 19 SERVICE
    "link",  # 20 SERVICE
    "remote_job",  # 21 SERVICE
    "gopher",  # 22 SERVICE
    "ssh",  # 23 SERVICE
    "name",  # 24 SERVICE
    "whois",  # 25 SERVICE
    "domain",  # 26 SERVICE
    "login",  # 27 SERVICE
    "imap4",  # 28 SERVICE
    "daytime",  # 29 SERVICE
    "ctf",  # 30 SERVICE
    "nntp",  # 31 SERVICE
    "shell",  # 32 SERVICE
    "IRC",  # 33 SERVICE
    "nnsp",  # 34 SERVICE
    "http_443",  # 35 SERVICE
    "exec",  # 36 SERVICE
    "printer",  # 37 SERVICE
    "efs",  # 38 SERVICE
    "courier",  # 39 SERVICE
    "uucp",  # 40 SERVICE
    "klogin",  # 41 SERVICE
    "kshell",  # 42 SERVICE
    "echo",  # 43 SERVICE
    "discard",  # 44 SERVICE
    "systat",  # 45 SERVICE
    "supdup",  # 46 SERVICE
    "iso_tsap",  # 47 SERVICE
    "hostnames",  # 48 SERVICE
    "csnet_ns",  # 49 SERVICE
    "pop_2",  # 50 SERVICE
    "sunrpc",  # 51 SERVICE
    "uucp_path",  # 52 SERVICE
    "netbios_ns",  # 53 SERVICE
    "netbios_ssn",  # 54 SERVICE
    "netbios_dgm",  # 55 SERVICE
    "sql_net",  # 56 SERVICE
    "vmnet",  # 57 SERVICE
    "bgp",  # 58 SERVICE
    "Z39_50",  # 59 SERVICE
    "ldap",  # 60 SERVICE
    "netstat",  # 61 SERVICE
    "urh_i",  # 62 SERVICE
    "X11",  # 63 SERVICE
    "urp_i",  # 64 SERVICE
    "pm_dump",  # 65 SERVICE
    "tftp_u",  # 66 SERVICE
    "tim_i",  # 67 SERVICE
    "red_i",  # 68 SERVICE
    "icmp",  # 69 SERVICE
    "http_2784",  # 70 SERVICE
    "harvest",  # 71 SERVICE
    "aol",  # 72 SERVICE
    "http_8001",  # 73 SERVICE
    "OTH",  # 74 FLAG
    "RSTOS0",  # 75 FLAG
    "SF",  # 76 FLAG
    "SH",  # 77 FLAG
    "RSTO",  # 78 FLAG
    "S2",  # 79 FLAG
    "S1",  # 80 FLAG
    "REJ",  # 81 FLAG
    "S3",  # 82 FLAG
    "RSTR",  # 83 FLAG
    "S0",  # 84 FLAG
    "src_bytes",  # 85
    "dst_bytes",  # 86
    "land",  # 87
    "wrong_fragment",  # 88
    "urgent",  # 89
    "hot",  # 90
    "num_failed_logins",  # 91
    "logged_in",  # 92
    "num_compromised",  # 93
    "root_shell",  # 94
    "su_attempted",  # 95
    "num_root",  # 96
    "num_file_creations",  # 97
    "num_shells",  # 98
    "num_access_files",  # 99
    "is_host_login",  # 100
    "is_guest_login",  # 101
    "count",  # 102
    "srv_count",  # 103
    "serror_rate",  # 104
    "srv_serror_rate",  # 105
    "rerror_rate",  # 106
    "srv_rerror_rate",  # 107
    "same_srv_rate",  # 108
    "diff_srv_rate",  # 109
    "srv_diff_host_rate",  # 110
    "dst_host_count",  # 111
    "dst_host_srv_count",  # 112
    "dst_host_same_srv_rate",  # 113
    "dst_host_diff_srv_rate",  # 114
    "dst_host_same_src_port_rate",  # 115
    "dst_host_srv_diff_host_rate",  # 116
    "dst_host_serror_rate",  # 117
    "dst_host_srv_serror_rate",  # 118
    "dst_host_rerror_rate",  # 119
    "dst_host_srv_rerror_rate",  # 120
    "difficulty",  # 121
]

idxs = [
    0,
    10, 102, 91, 59, 40, 79, 121, 108, 107, 120, 82,
    15, 115, 110, 84, 89, 53, 44, 48, 26, 58, 6,
    14, 1, 28, 32, 52, 57, 16, 56, 37, 30, 5,
    42, 55, 34, 76, 94, 97, 96, 80, 95, 9, 17,
    72, 75, 73, 99, 88, 92, 98, 83, 38, 100, 41,
    64, 33, 66, 101, 68, 71, 63, 74, 69, 67, 90,
    3, 65, 62, 78, 18, 22, 51, 12, 81, 20, 60,
    7, 36, 45, 50, 24, 54, 43, 25, 39, 19, 29,
    8, 27, 31, 47, 61, 46, 13, 21, 49, 23, 35,
    119, 85, 106, 105, 118, 103, 104, 86, 117, 11, 111,
    2, 112, 113, 109, 77, 114, 93, 4, 87, 116, 70,
]

# idxs = [
#     0,
#     116, 40, 5, 6, 31, 79, 36, 35, 7, 3, 29,
#     111, 70, 11, 117, 13, 21, 19, 32, 47, 42, 89,
#     27, 52, 16, 12, 76, 57, 78, 61, 110, 115, 15,
#     60, 41, 24, 80, 38, 64, 34, 81, 84, 1, 14,
#     33, 73, 88, 69, 72, 68, 101, 74, 71, 90, 92,
#     26, 75, 83, 63, 67, 66, 99, 98, 100, 94, 97,
#     50, 28, 62, 18, 22, 51, 96, 95, 9, 17, 65,
#     53, 48, 56, 44, 55, 43, 54, 20, 37, 25, 58,
#     10, 102, 91, 46, 49, 8, 108, 107, 120, 121, 82,
#     85, 106, 119, 105, 30, 45, 59, 23, 86, 87, 39,
#     118, 103, 104, 112, 2, 113, 109, 77, 114, 93, 4,
# ]

# idxs = [
#     0,
#     110, 115, 14, 13, 70, 116, 93, 114, 77, 109, 113,
#     8, 58, 31, 40, 27, 78, 87, 86, 4, 112, 2,
#     1, 91, 95, 25, 54, 47, 36, 21, 89, 104, 103,
#     106, 85, 118, 5, 32, 42, 61, 66, 62, 24, 48,
#     105, 119, 6, 39, 46, 100, 94, 101, 57, 43, 60,
#     80, 22, 97, 90, 98, 99, 96, 76, 71, 12, 26,
#     53, 88, 75, 74, 67, 69, 72, 73, 63, 68, 18,
#     17, 16, 33, 38, 51, 49, 56, 50, 7, 3, 15,
#     28, 64, 55, 34, 65, 35, 82, 121, 108, 120, 107,
#     111, 29, 52, 45, 37, 30, 81, 20, 23, 84, 59,
#     10, 102, 79, 41, 44, 83, 19, 9, 92, 117, 11,
# ]

# COLUMNS = [BASE_COLUMNS[i] for i in idxs]
COLUMNS = BASE_COLUMNS

[
    # (1)
    "duration",
    # (2)
    "src_bytes",
    "dst_bytes",
    # (4)
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    # (5)
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    # (4)
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    # (2)
    "is_host_login",
    "is_guest_login",
    # (2)
    "count",
    "srv_count",
    # (7)
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    # (10)
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    # SERVICE (71)
    "http",
    "smtp",
    "finger",
    "domain_u",
    "auth",
    "telnet",
    "ftp",
    "eco_i",
    "ntp_u",
    "ecr_i",
    "other",
    "private",
    "pop_3",
    "ftp_data",
    "rje",
    "time",
    "mtp",
    "link",
    "remote_job",
    "gopher",
    "ssh",
    "name",
    "whois",
    "domain",
    "login",
    "imap4",
    "daytime",
    "ctf",
    "nntp",
    "shell",
    "IRC",
    "nnsp",
    "http_443",
    "exec",
    "printer",
    "efs",
    "courier",
    "uucp",
    "klogin",
    "kshell",
    "echo",
    "discard",
    "systat",
    "supdup",
    "iso_tsap",
    "hostnames",
    "csnet_ns",
    "pop_2",
    "sunrpc",
    "uucp_path",
    "netbios_ns",
    "netbios_ssn",
    "netbios_dgm",
    "sql_net",
    "vmnet",
    "bgp",
    "Z39_50",
    "ldap",
    "netstat",
    "urh_i",
    "X11",
    "urp_i",
    "pm_dump",
    "tftp_u",
    "tim_i",
    "red_i",
    "icmp",
    "http_2784",
    "harvest",
    "aol",
    "http_8001",
    # PROTOCOL (2)
    "tcp",
    "udp",
    # FLAG (11)
    "OTH",
    "RSTOS0",
    "SF",
    "SH",
    "RSTO",
    "S2",
    "S1",
    "REJ",
    "S3",
    "RSTR",
    "S0",
]

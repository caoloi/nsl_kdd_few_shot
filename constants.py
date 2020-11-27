# REFERENCES PARAMETERS
CONFIG = {
    "epochs": 50,
    "batch_size": 128,
    "num_classes": 5,
    # "num_models": 1,
    "num_models": 6,
    "dataset": "kdd",
    "train_data": "train+",
    "img_rows": 11,
    "img_cols": 11,
    "output_dim": 121,
    "process_num": 6,
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

SAMPLE_NUM_PER_LABEL = {
    # normal
    "normal.": [67343 * 0 + 12000, 9711 * 0 + 10],  # 0
    # probe
    "ipsweep.": [3599 * 0, 141 * 0 + 2],  # 1
    "nmap.": [1493 * 0, 73 * 0 + 0],  # 2
    "portsweep.": [2931 * 0, 157 * 0 + 2],  # 3
    "satan.": [3633 * 0, 735 * 0 + 2],  # 4
    "saint.": [0 * 0, 319 * 0 + 2],  # 5
    "mscan.": [0 * 0, 996 * 0 + 2],  # 6
    # dos
    "back.": [956 * 0 + 3000, 359 * 0 + 2],  # 7
    "land.": [18 * 0, 7 * 0 + 0],  # 8
    "neptune.": [41214 * 0 + 1305 * 0 + 3000, 4657 * 0 + 2],  # 9
    "pod.": [201 * 0, 41 * 0 + 0],  # 10
    "smurf.": [2646 * 0 + 3000, 665 * 0 + 2],  # 11
    "teardrop.": [892 * 0 + 3000, 12 * 0 + 0],  # 12
    "apache2.": [0 * 0, 737 * 0 + 2],  # 13
    "udpstorm.": [0 * 0, 2 * 0 + 0],  # 14
    "processtable.": [0 * 0, 685 * 0 + 2],  # 15
    "mailbomb.": [0 * 0, 293 * 0 + 0],  # 16
    # u2r
    "buffer_overflow.": [30 * 0, 20 * 0 + 2],  # 17
    "loadmodule.": [9 * 0, 2 * 0 + 0],  # 18
    "perl.": [3 * 0, 2 * 0 + 0],  # 19
    "rootkit.": [10 * 0, 13 * 0 + 2],  # 20
    "xterm.": [0 * 0, 13 * 0 + 2],  # 21
    "ps.": [0 * 0, 15 * 0 + 2],  # 22
    "sqlattack.": [0 * 0, 2 * 0 + 0],  # 23
    "httptunnel.": [0 * 0, 133 * 0 + 2],  # 24
    # r2l
    # "httptunnel.": [0 * 0, 133 * 0 + 2],  # 24
    "ftp_write.": [8 * 0, 3 * 0 + 0],  # 25
    "guess_passwd.": [53 * 0 + 3000, 1231 * 0 + 2],  # 26
    "imap.": [11 * 0 + 3000, 1 * 0 + 0],  # 27
    "multihop.": [7 * 0, 18 * 0 + 1],  # 28
    "phf.": [4 * 0, 2 * 0 + 0],  # 29
    "spy.": [2 * 0, 0 * 0 + 0],  # 30
    "warezclient.": [890 * 0 + 3000, 0 * 0 + 0],  # 31
    "warezmaster.": [20 * 0 + 3000, 944 * 0 + 2],  # 32
    "snmpgetattack.": [0 * 0, 178 * 0 + 2],  # 33
    "named.": [0 * 0, 17 * 0 + 1],  # 34
    "xlock.": [0 * 0, 9 * 0 + 0],  # 35
    "xsnoop.": [0 * 0, 4 * 0 + 0],  # 36
    "sendmail.": [0 * 0, 14 * 0 + 0],  # 37
    "worm.": [0 * 0, 2 * 0 + 0],  # 38
    "snmpguess.": [0 * 0, 331 * 0 + 2],  # 39
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
    "label",
    "duration",
    "tcp",
    "udp",
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
    "difficulty",
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

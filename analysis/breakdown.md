# Breakdown of force questions (seed 0, full data)

## YLM + MiniLM

### test: 5717 force questions

- attention peak on the asked atom: 0.746
- peak on the asked atom or a bonded neighbour: 0.952
- mean share of attention on the asked atom: 0.745
- force MAE: 0.0074

| group | n | peak on atom | peak on atom or neighbour | attention share | force MAE |
|---|---|---|---|---|---|
| 13-18 atoms | 3091 | 0.765 | 0.961 | 0.764 | 0.0075 |
| <=12 atoms | 179 | 0.732 | 0.894 | 0.730 | 0.0110 |
| >18 atoms | 2447 | 0.723 | 0.946 | 0.721 | 0.0069 |
| element C | 3491 | 0.983 | 0.987 | 0.980 | 0.0071 |
| element F | 7 | 0.000 | 0.571 | 0.000 | 0.0106 |
| element H | 443 | 0.000 | 0.982 | 0.000 | 0.0046 |
| element N | 753 | 0.992 | 0.993 | 0.992 | 0.0087 |
| element O | 1023 | 0.085 | 0.796 | 0.085 | 0.0085 |
| no ring words | 5501 | 0.755 | 0.962 | 0.753 | 0.0069 |
| ring words | 216 | 0.528 | 0.704 | 0.522 | 0.0205 |

Atom descriptions with the lowest attention accuracy (>= 20 questions):

| description | n | atom acc |
|---|---|---|
| oxygen atom in a ring bonded to two carbons | 20 | 0.000 |
| oxygen atom not in a ring bonded to two carbons | 22 | 0.000 |
| oxygen atom bonded to one carbon and one nitrogen | 24 | 0.000 |
| hydrogen atom bonded to one nitrogen | 171 | 0.000 |
| hydrogen atom bonded to one oxygen | 264 | 0.000 |
| oxygen atom bonded to one carbon | 293 | 0.000 |
| oxygen atom bonded to two carbons | 399 | 0.000 |
| carbon atom in a ring bonded to two hydrogens and two carbons | 30 | 0.033 |
| oxygen atom bonded to one hydrogen and one carbon | 247 | 0.279 |
| carbon atom bonded to one carbon and two nitrogens | 55 | 0.982 |

### test_heldout: 5717 force questions

- attention peak on the asked atom: 0.700
- peak on the asked atom or a bonded neighbour: 0.912
- mean share of attention on the asked atom: 0.695
- force MAE: 0.0128

| group | n | peak on atom | peak on atom or neighbour | attention share | force MAE |
|---|---|---|---|---|---|
| 13-18 atoms | 3091 | 0.710 | 0.916 | 0.706 | 0.0139 |
| <=12 atoms | 179 | 0.665 | 0.844 | 0.660 | 0.0163 |
| >18 atoms | 2447 | 0.689 | 0.913 | 0.682 | 0.0113 |
| element C | 3491 | 0.896 | 0.940 | 0.888 | 0.0145 |
| element F | 7 | 0.000 | 0.286 | 0.000 | 0.0167 |
| element H | 443 | 0.000 | 0.982 | 0.000 | 0.0058 |
| element N | 753 | 0.979 | 0.980 | 0.978 | 0.0099 |
| element O | 1023 | 0.132 | 0.740 | 0.132 | 0.0123 |
| no ring words | 5501 | 0.708 | 0.922 | 0.703 | 0.0124 |
| ring words | 216 | 0.495 | 0.657 | 0.468 | 0.0233 |

## YLM + MatSciBERT

### test: 5717 force questions

- attention peak on the asked atom: 0.942
- peak on the asked atom or a bonded neighbour: 0.982
- mean share of attention on the asked atom: 0.941
- force MAE: 0.0068

| group | n | peak on atom | peak on atom or neighbour | attention share | force MAE |
|---|---|---|---|---|---|
| 13-18 atoms | 3091 | 0.953 | 0.984 | 0.952 | 0.0070 |
| <=12 atoms | 179 | 0.978 | 0.978 | 0.976 | 0.0090 |
| >18 atoms | 2447 | 0.924 | 0.980 | 0.924 | 0.0064 |
| element C | 3491 | 0.983 | 0.988 | 0.982 | 0.0072 |
| element F | 7 | 0.571 | 0.857 | 0.571 | 0.0102 |
| element H | 443 | 0.404 | 0.880 | 0.401 | 0.0039 |
| element N | 753 | 0.995 | 0.995 | 0.993 | 0.0083 |
| element O | 1023 | 0.997 | 0.997 | 0.996 | 0.0056 |
| no ring words | 5501 | 0.950 | 0.989 | 0.949 | 0.0064 |
| ring words | 216 | 0.727 | 0.806 | 0.726 | 0.0167 |

Atom descriptions with the lowest attention accuracy (>= 20 questions):

| description | n | atom acc |
|---|---|---|
| carbon atom in a ring bonded to two hydrogens and two carbons | 30 | 0.000 |
| hydrogen atom bonded to one oxygen | 264 | 0.000 |
| nitrogen atom bonded to one carbon and one oxygen | 29 | 0.931 |
| carbon atom bonded to three carbons and one nitrogen | 70 | 0.986 |
| oxygen atom bonded to one hydrogen and one carbon | 247 | 0.992 |
| oxygen atom in a ring bonded to two carbons | 20 | 1.000 |
| carbon atom bonded to two nitrogens and one oxygen | 22 | 1.000 |
| oxygen atom not in a ring bonded to two carbons | 22 | 1.000 |
| carbon atom bonded to one hydrogen and two nitrogens | 24 | 1.000 |
| oxygen atom bonded to one carbon and one nitrogen | 24 | 1.000 |

### test_heldout: 5717 force questions

- attention peak on the asked atom: 0.920
- peak on the asked atom or a bonded neighbour: 0.968
- mean share of attention on the asked atom: 0.914
- force MAE: 0.0087

| group | n | peak on atom | peak on atom or neighbour | attention share | force MAE |
|---|---|---|---|---|---|
| 13-18 atoms | 3091 | 0.931 | 0.972 | 0.925 | 0.0090 |
| <=12 atoms | 179 | 0.933 | 0.944 | 0.933 | 0.0129 |
| >18 atoms | 2447 | 0.904 | 0.965 | 0.898 | 0.0080 |
| element C | 3491 | 0.957 | 0.981 | 0.948 | 0.0093 |
| element F | 7 | 0.286 | 0.429 | 0.295 | 0.0119 |
| element H | 443 | 0.402 | 0.837 | 0.399 | 0.0043 |
| element N | 753 | 0.959 | 0.959 | 0.957 | 0.0116 |
| element O | 1023 | 0.992 | 0.992 | 0.991 | 0.0065 |
| no ring words | 5501 | 0.929 | 0.976 | 0.922 | 0.0083 |
| ring words | 216 | 0.694 | 0.764 | 0.698 | 0.0187 |

## task ID (no text)

### test: 5717 force questions

- attention peak on the asked atom: 0.009
- peak on the asked atom or a bonded neighbour: 0.129
- mean share of attention on the asked atom: 0.010
- force MAE: 0.0417

| group | n | peak on atom | peak on atom or neighbour | attention share | force MAE |
|---|---|---|---|---|---|
| 13-18 atoms | 3091 | 0.013 | 0.139 | 0.014 | 0.0430 |
| <=12 atoms | 179 | 0.022 | 0.151 | 0.023 | 0.0423 |
| >18 atoms | 2447 | 0.004 | 0.114 | 0.004 | 0.0398 |
| element C | 3491 | 0.000 | 0.163 | 0.000 | 0.0456 |
| element F | 7 | 0.000 | 0.000 | 0.031 | 0.0222 |
| element H | 443 | 0.122 | 0.122 | 0.128 | 0.0252 |
| element N | 753 | 0.000 | 0.088 | 0.000 | 0.0439 |
| element O | 1023 | 0.000 | 0.048 | 0.000 | 0.0336 |
| no ring words | 5501 | 0.010 | 0.130 | 0.010 | 0.0416 |
| ring words | 216 | 0.000 | 0.106 | 0.000 | 0.0439 |

### test_heldout: 5717 force questions

- attention peak on the asked atom: 0.009
- peak on the asked atom or a bonded neighbour: 0.129
- mean share of attention on the asked atom: 0.010
- force MAE: 0.0417

| group | n | peak on atom | peak on atom or neighbour | attention share | force MAE |
|---|---|---|---|---|---|
| 13-18 atoms | 3091 | 0.013 | 0.139 | 0.014 | 0.0430 |
| <=12 atoms | 179 | 0.022 | 0.151 | 0.023 | 0.0423 |
| >18 atoms | 2447 | 0.004 | 0.114 | 0.004 | 0.0398 |
| element C | 3491 | 0.000 | 0.163 | 0.000 | 0.0456 |
| element F | 7 | 0.000 | 0.000 | 0.031 | 0.0222 |
| element H | 443 | 0.122 | 0.122 | 0.128 | 0.0252 |
| element N | 753 | 0.000 | 0.088 | 0.000 | 0.0439 |
| element O | 1023 | 0.000 | 0.048 | 0.000 | 0.0336 |
| no ring words | 5501 | 0.010 | 0.130 | 0.010 | 0.0416 |
| ring words | 216 | 0.000 | 0.106 | 0.000 | 0.0439 |


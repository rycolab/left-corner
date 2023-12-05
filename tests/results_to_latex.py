import pandas as pd, dill, warnings
from path import Path
from leftcorner.misc import colors

warnings.filterwarnings("ignore")

outputs = Path('./output')
out_spmrl = outputs / 'spmrl'


languages = [
    'Basque',
    'English',
    'French',
    'German',
    'Hebrew',
    'Hungarian',
    'Korean',
    'Polish',
    'Swedish',
]

def load_data(fname):
    obj = None
    with open(fname, 'rb') as f:
        obj = dill.load(f)
    return obj

def initial_info(columns):
    return {col:[] for col in columns}

def create_info_dict(include_ratio, langs=languages):
    info = initial_info('Language Method Raw Trim Eps'.split())
    methods = ['SLCT', 'GLCT']
    if include_ratio: methods.append('SLCT/GLCT')
    info['Method'] = [x for x in range(len(langs)) for x in methods]
    return info

def get_table(mode, langs=languages, include_ratio=True):
    # options for mode: size, num_rules, time, slides
    if mode == 'time':
        info = initial_info('Language SLCT GLCT SLCT/GLCT'.split())
    elif mode == 'slides':
        info = initial_info(['Language'] + langs)
        info['Language'].extend(['Original', 'SLCT', 'GLCT'])
    else:
        info = create_info_dict(include_ratio)

    for language in langs:
        output_dir = outputs / 'atis' if language == 'English' else out_spmrl / language

        results = load_data(output_dir / 'results.pkl')
        data = results['size'] if mode == 'slides' else results[mode]

        if mode == 'time':
            info['Language'].append(language)
            slct_sum = data['slct'] + data['t_slct'] + data['e_slct']
            info['SLCT'].append(f'{slct_sum:.2f}')
            glct_sum = data['glct'] + data['t_glct'] + data['e_glct']
            info['GLCT'].append(f'{glct_sum:.2f}')
            info['SLCT/GLCT'].append(f'{slct_sum/glct_sum:.2f}')
        elif mode == 'slides':
            info[language].extend((f'{data["cfg"]:,}', f'{data["slct"]:,}', f'{data["glct"]:,}'))
        else:
            info['Language'].extend((language, f'({data["cfg"]:,})'))
            info['Raw'].extend((f'{data["slct"]:,}', f'{data["glct"]:,}'))
            info['Trim'].extend((f'{data["t_slct"]:,}', f'{data["t_glct"]:,}'))
            info['Eps'].extend((f'{data["e_slct"]:,}', f'{data["e_glct"]:,}'))
            if include_ratio:
                info['Language'].append('')
                info['Raw'].append(f'{data["slct"]/data["glct"]:.2f}')
                info['Trim'].append(f'{data["t_slct"]/data["t_glct"]:.2f}')
                info['Eps'].append(f'{data["e_slct"]/data["e_glct"]:.2f}')

    df = pd.DataFrame(info)
    if mode == 'time' and not include_ratio:
        ltable = df.to_latex(index=False, index_names=False, columns=columns[:-1])
    else:
        ltable = df.to_latex(index=False, index_names=False)
    print(ltable, '\n')

get_table('size', include_ratio = False)
get_table('size')
get_table('num_rules')
get_table('time')
get_table('slides', langs=languages[:5])
get_table('slides', langs=languages[5:])

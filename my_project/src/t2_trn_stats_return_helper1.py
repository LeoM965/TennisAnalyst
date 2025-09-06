import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def extract_year(match_str):
    match = re.search(r'\b(20\d{2})\b', match_str)
    return int(match.group(1)) if match else None


def learn_return_weights(df):
    pca = PCA(n_components=1)
    scaler = StandardScaler()

    return_cols = ['Overall_RiP', 'Overall_RiP_W', 'Overall_RetWnr']
    available_cols = [col for col in return_cols if col in df.columns]

    if len(available_cols) > 0:
        return_data = df[available_cols].dropna()
        if len(return_data) > 0:
            scaled_data = scaler.fit_transform(return_data)
            pca.fit(scaled_data)
            return_weights = abs(pca.components_[0])
            return_weights = return_weights / return_weights.sum()
        else:
            return_weights = [0.4, 0.4, 0.2]
    else:
        return_weights = [0.4, 0.4, 0.2]

    first_second_cols = ['First_RiP', 'Second_RiP']
    available_serve_cols = [col for col in first_second_cols if col in df.columns]

    if len(available_serve_cols) == 2:
        serve_data = df[available_serve_cols].dropna()
        if len(serve_data) > 0:
            first_importance = serve_data['First_RiP'].std()
            second_importance = serve_data['Second_RiP'].std()
            total_importance = first_importance + second_importance + 0.001
            serve_weights = [first_importance / total_importance, second_importance / total_importance]
        else:
            serve_weights = [0.6, 0.4]
    else:
        serve_weights = [0.6, 0.4]

    return return_weights, serve_weights
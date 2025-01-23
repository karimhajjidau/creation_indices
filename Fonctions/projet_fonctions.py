import pandas as pd
from typing import Dict, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.figure import Figure
from typing import Callable
import os
from fpdf import FPDF
import plotly.express as px
import plotly.graph_objects as go
import kaleido
import smtplib
from email.message import EmailMessage


def process_price_data(file_path, sheet_name, start_date, end_date = "2022-10-22" , threshold_ratio=0.95):
    """Met les prix dans un data-frame et traite les valeurs manquantes"""
    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=1).loc[start_date:end_date, :]

    # Seuil pour conserver uniquement les titres avec suffisamment de données non N/A
    threshold = threshold_ratio * len(df)
    df = df.dropna(axis=1, thresh=threshold)

    # Interpolation des valeurs manquantes pour lisser les données
    df = df.interpolate()
    return df

def convert_to_currency(df, forex_df, base_currency='EUR', target_currency='USD'):
    """Convertit un data-frame de prix dans une autre monnaie."""
    forex_column = f"{base_currency}{target_currency}"
    df[forex_column] = df.index.map(lambda date: forex_df.loc[date, forex_column])
    columns_to_convert = df.columns.difference([forex_column])
    df_converted = df.copy()
    df_converted[columns_to_convert] = df[columns_to_convert].mul(df[forex_column], axis=0)
    return df_converted

def process_qualitative_data(file_path, sheet_name, df_prices, df_forex, date, df_SXXP):
    """Traite un data-frame de données qualitatives."""
    def ajuster_dividendes(yield_, freq):
        if freq == "Annual":
            return yield_
        elif freq == "Semi-Anl":
            return yield_ * 2
        elif freq == "Quarterly":
            return yield_ * 4
        else:
            return yield_

    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=0).dropna()

    df = df[df.index.isin(df_prices.columns)]
    
    df['Currency'] = ['EUR' if idx in df_SXXP.columns else 'USD' for idx in df.index]
    df['EQY_DVD_YLD_Adjusted'] = df.apply(
        lambda row: ajuster_dividendes(row['EQY_DVD_YLD_IND'], row['DVD_FREQ']), axis=1
    )
    cols_to_convert = ['CUR_MKT_CAP', 'EQY_SH_OUT', 'PX_LAST']
    for col in cols_to_convert:
        if col in df.columns:
            df.loc[df['Currency'] == 'EUR', col] *= df_forex.loc[date,"EURUSD"]
    
    colonnes_a_convertir = ['PX_TO_BOOK_RATIO', 'PE_RATIO', 'EQY_DVD_YLD_IND']
    for col in colonnes_a_convertir:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convertir en float

    return df
    
    


def style_value(df: pd.DataFrame, *args) -> pd.DataFrame:
    """Stratégie Value basée sur le Price-to-Book, le PE ratio et le Dividend Yield."""
    df['Z_PX_TO_BOOK'] = (1 / df['PX_TO_BOOK_RATIO'] - (1 / df['PX_TO_BOOK_RATIO']).mean()) / (1 / df['PX_TO_BOOK_RATIO']).std()
    df['Z_PE_RATIO'] = (1 / df['PE_RATIO'] - (1 / df['PE_RATIO']).mean()) / (1 / df['PE_RATIO']).std()
    df['Z_EQY_DVD_YLD'] = (df['EQY_DVD_YLD_Adjusted'] - df['EQY_DVD_YLD_Adjusted'].mean()) / df['EQY_DVD_YLD_Adjusted'].std()
    df['Value_Z_Score'] = df[['Z_PX_TO_BOOK', 'Z_PE_RATIO', 'Z_EQY_DVD_YLD']].mean(axis=1)
    return df.sort_values(by='Value_Z_Score', ascending=False)

def style_momentum_sector(df_qual: pd.DataFrame, df_prices: pd.DataFrame, df_secteurs: pd.DataFrame, bics_criteria: list) -> pd.DataFrame:
    """Stratégie Momentum sectoriel, prenant en compte 12 mois précédents, et les secteurs BICS désirés"""
    common_stocks = df_qual.index.intersection(df_prices.columns).intersection(df_secteurs.index)
    df_qual = df_qual.loc[common_stocks]
    df_prices = df_prices[common_stocks]
    df_secteurs = df_secteurs.loc[common_stocks]
    df_qual = df_qual[df_secteurs['BICS_LEVEL_3_INDUSTRY_NAME'].isin(bics_criteria)]
    df_qual['Momentum'] = df_prices.pct_change(periods=12).sum(axis=0)  # 12 mois précédents
    return df_qual.sort_values(by='Momentum', ascending=False)

def style_high_div_low_vol(df_qual: pd.DataFrame, df_prices: pd.DataFrame) -> pd.DataFrame:
    """Stratégie High Dividend Low Volatility, avec volatilité sur 12 mois."""
    df_qual['Volatility'] = df_prices.pct_change().rolling(window=12).std().iloc[-1] # 12 mois de volatilité
    df_qual['Z_EQY_DVD_YLD'] = (df_qual['EQY_DVD_YLD_Adjusted'] - df_qual['EQY_DVD_YLD_Adjusted'].mean()) / df_qual['EQY_DVD_YLD_Adjusted'].std()
    df_qual['Z_Volatility'] = (df_qual['Volatility'].mean() - df_qual['Volatility']) / df_qual['Volatility'].std()
    df_qual['Dividend_LowVol_Score'] = df_qual[['Z_EQY_DVD_YLD', 'Z_Volatility']].mean(axis=1)
    return df_qual.sort_values(by='Dividend_LowVol_Score', ascending=False)

def filtrer_et_selectionner(df_qual: pd.DataFrame, df_prices: pd.DataFrame, style_function: Callable, bics_criteria=[], df_secteurs = [], n_actions=50) -> pd.DataFrame:
    """Applique un style et sélectionne les n meilleures actions après tri."""
    if style_function == style_momentum_sector:
        df_selected = style_function(df_qual, df_prices, df_secteurs, bics_criteria).head(n_actions)
    else :
        df_selected = style_function(df_qual, df_prices).head(n_actions)
    df_selected['Weight'] = df_selected['CUR_MKT_CAP'] / df_selected['CUR_MKT_CAP'].sum()
    return df_selected

def calculer_indice(df_selected: pd.DataFrame, df_prices: pd.DataFrame, rebalance_date,base=100) -> pd.Series:
    """Calcule un indice pondéré des prix en base 100."""
    prices = df_prices[df_selected.index]
    normalized_prices = prices / prices.loc[rebalance_date] * base
    return normalized_prices.mul(df_selected['Weight'], axis=1).sum(axis=1)

def rebalancer_indice(df_prices: pd.DataFrame, df_qual_dict: dict, df_secteurs: pd.DataFrame, rebalance_dates: list, style_function: Callable, bics_criteria = [], n_actions=50, base=100):
    """Rebalancement d'un indice en fonction d'une stratégie, avec alignement des dates et affichage par période de rebalancement."""
    indices_rebalances = {}
    last_value = base
    start_date = df_prices.index[0]  # Première date disponible
    
    for i, rebalance_date in enumerate(rebalance_dates):
        next_rebalance_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else df_prices.index[-1]
        df_selected = filtrer_et_selectionner(df_qual_dict[rebalance_date], df_prices.loc[start_date:next_rebalance_date], style_function, bics_criteria, df_secteurs,n_actions=n_actions)
        indice_temp = calculer_indice(df_selected, df_prices, rebalance_date, base=last_value)
        indices_rebalances[rebalance_date] = {
            'indice': indice_temp,
            'company weights': df_selected['Weight']
        }
        last_value = indice_temp.loc[next_rebalance_date]
    
    return indices_rebalances

def tracer_indices_comparaison(indices_rebalances, titre, debut="2019-01-01", fin="2021-01-01", save_path=None, pdf=None):
    """Trace les indices rebalancés à chaque date"""
    plt.figure(figsize=(12, 6))
    for date, data in indices_rebalances.items():
        indice = data['indice']
        indice = indice[debut:fin]
        plt.plot(indice, linestyle='dashed', label=f"{titre} rebalancé le {date}")
    
    plt.title(f"Comparaison des Indices {titre} à partir de " + debut)
    plt.xlabel("Date")
    plt.ylabel("Valeur de l'Indice (Base 100)")
    plt.legend()
    plt.grid(True)
    if save_path:
        filename = f"{save_path}/comparaison_{titre}.png"
        plt.savefig(filename)
        if pdf:
            pdf.image(filename, x=10, w=100)
    else:
        plt.show()


def calculer_performance_risque(indice, indice_ref, debut, fin):
    """Calcule les métriques de performance et de risque pour un indice donné."""

    indice = indice[debut:fin]
    indice_ref = indice_ref[debut:fin]

    # Performance totale
    performance_totale = (indice.iloc[-1] / indice.iloc[0]) - 1

    # Performance annualisée
    nb_annees = (indice.index[-1] - indice.index[0]).days / 365.25
    performance_annuelle = (1 + performance_totale) ** (1 / nb_annees) - 1

     # Rendements journaliers
    rendements = indice.pct_change().dropna()
    rendements_ref = indice_ref.pct_change().dropna()

    # Aligner les rendements après changement de pourcentage
    rendements, rendements_ref = rendements.align(rendements_ref, join='inner')
    rendements_ref = rendements_ref.squeeze()
    volatilite = rendements.std() * np.sqrt(252)

    # Max Drawdown
    cumul = (1 + rendements).cumprod()
    max_drawdown = (cumul / cumul.cummax() - 1).min()

    # Beta
    covariance_matrix = np.cov(rendements, rendements_ref)
    if covariance_matrix.shape == (2,2):  # Vérifier que la covariance est bien calculée
        beta = covariance_matrix[0, 1] / rendements_ref.var()
    else:
        beta = np.nan  # Si le calcul de la covariance échoue

    # Alpha
    alpha = performance_annuelle - beta * rendements_ref.mean() * 252

    # Ratio de Sharpe (taux sans risque = 2%)
    taux_sans_risque = 0.02
    ratio_sharpe = (rendements.mean() * 252 - taux_sans_risque) / volatilite

    return {
        "Performance Totale": performance_totale,
        "Performance Annualisée": performance_annuelle,
        "Volatilité": volatilite,
        "Max Drawdown": max_drawdown,
        "Beta": beta,
        "Alpha": alpha,
        "Ratio de Sharpe": ratio_sharpe
    }

def calculer_performances_benchmark(indice_ref, debut, fin):
    indice_ref = indice_ref[debut:fin]
    indice_ref = indice_ref.squeeze()

    # Performance totale
    performance_totale = (indice_ref.iloc[-1] / indice_ref.iloc[0]) - 1

    # Performance annualisée
    nb_annees = (indice_ref.index[-1] - indice_ref.index[0]).days / 365.25
    performance_annuelle = (1 + performance_totale) ** (1 / nb_annees) - 1

    # Performance annualisée
    nb_annees = (indice_ref.index[-1] - indice_ref.index[0]).days / 365.25

    # Rendements journaliers
    rendements_ref = indice_ref.pct_change().dropna()
    rendements_ref.name = None 

    # Aligner les rendements après changement de pourcentage
    volatilite = rendements_ref.std() * np.sqrt(252)

    # Max Drawdown
    cumul = (1 + rendements_ref).cumprod()
    max_drawdown = (cumul / cumul.cummax() - 1).min()

    beta = 1

    alpha = performance_annuelle - beta * rendements_ref.mean() * 252

    # Ratio de Sharpe (taux sans risque = 2%)
    taux_sans_risque = 0.02
    ratio_sharpe = (rendements_ref.mean() * 252 - taux_sans_risque) / volatilite

    return {
        "Performance Totale": performance_totale,
        "Performance Annualisée": performance_annuelle,
        "Volatilité": volatilite,
        "Max Drawdown": max_drawdown,
        "Beta": beta,
        "Alpha": alpha,
        "Ratio de Sharpe": ratio_sharpe
    }


    return


def plot_sector_piecharts(indices_rebalances, df_secteurs, strategy_name, top_n=5, save_path=None, pdf = None):
    """Génère un camembert interactif de répartition sectorielle avec Plotly, limitant le nombre de catégories affichées."""
    
    for date, data in indices_rebalances.items():
        df_weights = data['company weights']
        df_weights = df_weights.to_frame(name='Weight')
        df_weights['Sector'] = df_weights.index.map(lambda stock: df_secteurs.loc[stock, 'BICS_LEVEL_3_INDUSTRY_NAME'])
        
        df_sector_weights = df_weights.groupby('Sector').sum().reset_index()
        df_sector_weights = df_sector_weights.sort_values(by='Weight', ascending=False)
        
        if len(df_sector_weights) > top_n:
            df_sector_weights.loc[top_n:, 'Sector'] = 'Autres'
            df_sector_weights = df_sector_weights.groupby('Sector').sum().reset_index()
        
        fig = px.pie(df_sector_weights, values='Weight', names='Sector',
                     title=f"Répartition Sectorielle ({strategy_name} - {date})",
                     hole=0.3)

        if save_path:
            filename=f"{save_path}/sector_{date}.png"
            fig.write_image(filename)
            pdf.image(filename, x=10, w=100)
            pdf.ln(10)
        else:
            fig.show()
        

def plot_country_piecharts(indices_rebalances, df_qual_dict, strategy_name, top_n=5, save_path=None, pdf=None):
    """Génère un camembert interactif de répartition géographique avec Plotly, limitant le nombre de pays affichés."""
    
    for date, data in indices_rebalances.items():
        df_weights = data['company weights']
        df_weights = df_weights.to_frame(name='Weight')
        df_weights['COUNTRY'] = df_weights.index.map(lambda stock: df_qual_dict[date].loc[stock, 'COUNTRY'] if stock in df_qual_dict[date].index else "Unknown")
        
        df_country_weights = df_weights.groupby('COUNTRY').sum().reset_index()
        df_country_weights = df_country_weights.sort_values(by='Weight', ascending=False)
        
        if len(df_country_weights) > top_n:
            df_country_weights.loc[top_n:, 'COUNTRY'] = 'Autres'
            df_country_weights = df_country_weights.groupby('COUNTRY').sum().reset_index()
        
        fig = px.pie(df_country_weights, values='Weight', names='COUNTRY',
                     title=f"Répartition Géographique ({strategy_name} - {date})",
                     hole=0.3)

        if save_path:
            filename=f"{save_path}/country_{date}.png"
            fig.write_image(filename)
            pdf.image(filename, x=10, w=100)
            pdf.ln(10)
        else:
            fig.show()



class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, 'Rapport de Risque & Performance des Indices Rebalancés', border=False, ln=True, align='C')
        self.ln(5)
        self.set_draw_color(0, 0, 0)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

    def add_text(self, text):
        """Ajoute un texte propre avec un bon espacement."""
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 8, text)
        self.ln()

def save_plotly_table(dataframe, title, filename):
    """Créer un tableau avec Plotly, ajouter un titre et le sauvegarder en image."""

    # Colonnes à afficher en pourcentage
    percent_columns = ["Performance Totale", "Performance Annualisée", "Volatilité", "Max Drawdown", "Weight"]

    # Conversion des colonnes pertinentes en %
    for col in dataframe.columns:
        if col in percent_columns and dataframe[col].dtype in ['float64', 'float32']:
            dataframe[col] = (dataframe[col] * 100).round(2).astype(str) + " %"
        elif dataframe[col].dtype in ['float64', 'float32']:
            dataframe[col] = dataframe[col].round(2)

    # Création du tableau Plotly avec titre dans le layout
    fig = go.Figure()

    # Ajout du tableau
    fig.add_trace(go.Table(
        columnwidth=[150] + [100] * len(dataframe.columns),  # Largeur des colonnes
        header=dict(
            values=["<b>Indicateur</b>"] + [f"<b>{col}</b>" for col in dataframe.columns],
            fill_color='lightgrey',
            align='center',
            font=dict(color='black', size=12),
            height=30  # Espacement des en-têtes
        ),
        cells=dict(
            values=[dataframe.index] + [dataframe[col] for col in dataframe.columns],
            fill_color='white',
            align=['left'] + ['center'] * len(dataframe.columns),
            font=dict(color='black', size=11),
            height=25  # Espacement entre les cellules
        )
    ))

    # Ajout du titre via `layout.title`
    fig.update_layout(
        title=dict(
            text=f"<b>{title}</b>",  # Titre en gras
            x=0.5,  # Centrage horizontal
            y=0.97,  # Positionnement vertical proche du haut
            xanchor="center",
            yanchor="top",
            font=dict(size=16, color="black")
        ),
        margin=dict(l=10, r=10, t=80, b=10),  # Espace en haut pour le titre
        width=1000, height=500  # Taille de l'image
    )

    # Sauvegarde en image
    fig.write_image(filename, scale=2, width=1000, height=500)
    return filename

def creation_pdf(style, rebalance_dates, df_prices, df_qual_dict, df_benchmark, df_secteurs, bics_criteria = []):
    """Crée un pdf avec les performances"""
    
    indice = rebalancer_indice(df_prices, df_qual_dict, df_secteurs, rebalance_dates, style, bics_criteria, n_actions=50)
    resultats_performance = {
        "Indice non rebalancé": calculer_performance_risque(indice[rebalance_dates[0]]['indice'], df_benchmark, "2019-01-01", "2021-12-31"),
        "Indice Rebalancé au "+str(rebalance_dates[-1]): calculer_performance_risque(indice[rebalance_dates[-1]]['indice'], df_benchmark, "2019-01-01", "2021-12-31"),
        "Benchmark" : calculer_performances_benchmark(df_benchmark, "2019-01-01", "2021-12-31")
    }
    
    df_performance = pd.DataFrame(resultats_performance).T
    pdf = PDFReport()
    pdf.add_page()

    save_path = "."

    table_filename = "plotly_table.png"
    save_plotly_table(pd.DataFrame(resultats_performance).T, "Performances des indices rebalancés", table_filename)
    pdf.image(table_filename, x=10, y=pdf.get_y(), w=180)
    pdf.ln(60)

    plot_sector_piecharts(indice, df_secteurs, "Momentum Sectoriel", save_path=save_path, pdf=pdf)
    plot_country_piecharts(indice, df_qual_dict, "Momentum Sectoriel", save_path=save_path, pdf=pdf)

    # Tracer l'évolution des indices
    tracer_indices_comparaison(indice, "Style", "2019-01-01", "2021-12-31", save_path = save_path, pdf=pdf)


    # Récupérer les Top 10 entreprises pour 2018
    top10_2018_indices = indice[rebalance_dates[0]]['company weights'].nlargest(10)
    top10_2018 = top10_2018_indices.to_frame()
    top10_2018['Company Name'] = df_secteurs.loc[top10_2018_indices.index, 'NAME'].values

    # Récupérer les Top 10 entreprises pour 2020
    top10_2020_indices = indice[rebalance_dates[-1]]['company weights'].nlargest(10)
    top10_2020 = top10_2020_indices.to_frame()
    top10_2020['Company Name'] = df_secteurs.loc[top10_2020_indices.index, 'NAME'].values

    top10_2018["Company Name"] = top10_2018["Company Name"].astype(str)
    top10_2020["Company Name"] = top10_2020["Company Name"].astype(str)
    top10_2018["Weight"] = pd.to_numeric(top10_2018["Weight"], errors="coerce")
    top10_2020["Weight"] = pd.to_numeric(top10_2020["Weight"], errors="coerce")


    file_top10_2018 = "top10_2018.png"
    file_top10_2020 = "top10_2020.png"

    save_plotly_table(top10_2018, "Top 10 Entreprises - Indice Rebalancé 2018", file_top10_2018)
    save_plotly_table(top10_2020, "Top 10 Entreprises - Indice Rebalancé 2020", file_top10_2020)

    pdf.image(file_top10_2018, x=10, y=pdf.get_y(), w=190)
    pdf.ln(30)  # Espacement après le tableau

    pdf.image(file_top10_2020, x=10, y=pdf.get_y(), w=190) # Espacement après le tableau



    # Sauvegarder le PDF
    output_path = "Rapport_Risque_Performance.pdf"
    pdf.output(output_path)

    # Afficher le lien de téléchargement
    output_path


def envoi_mail(SMTP_SERVER = "smtp.gmail.com", SMTP_PORT = 587, EMAIL_SENDER = "karimhajji.kh@gmail.com", EMAIL_PASSWORD = "gfoghlppvdedzvak", EMAIL_RECEIVER = "karim.hajji@dauphine.eu", FILE_PATH = "Rapport_Risque_Performance.pdf"):
    """Création de l'e-mail"""
    msg = EmailMessage()
    msg["Subject"] = "Voici le fichier en pièce jointe"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.set_content("Bonjour,\n\nCi-joint le fichier demandé.\n\nCordialement.")

    try:
        with open(FILE_PATH, "rb") as file:
            file_data = file.read()
            file_name = os.path.basename(FILE_PATH)
            msg.add_attachment(file_data, maintype="application", subtype="octet-stream", filename=file_name)
    except FileNotFoundError:
        print(f"❌ Erreur : Le fichier '{FILE_PATH}' n'a pas été trouvé.")
        exit()


    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()  # Sécurise la connexion
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
            print("✅ E-mail envoyé avec succès !")
    except Exception as e:
        print(f"❌ Erreur lors de l'envoi : {e}")
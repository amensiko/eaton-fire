from statsmodels.tsa.stattools import ccf
import pandas as pd
import plotly.graph_objects as go

def load_wx():
    wx_daily = r"prism_wx/data/processed/daily_data.csv"
    df = pd.read_csv(wx_daily)
    print(df.head)
    #drop extraneous
    df.drop(columns=["Year", "Month", "Day"])
    #Convert to datetime
    df["DateString"] = pd.to_datetime(df["DateString"])

    return df


def cross_correlation(df):
    
    cross_correlation = ccf(x=df["ppt"], #x shifts in time, ie, for lag k, the data is shifted to start at x(t+k)
                            y=df["vpdmax"], #y stays in place
                            nlags=25, #More than this and things start to get unreliable (less data) - are there domain reasons to go beyond?
                            adjusted=True
                            )
    print(cross_correlation)
    
    return cross_correlation

def plot_correlations(cross_corr):

    #pull out the lag number (just the index of the correlation)
    lags = [i for i in range(len(cross_corr))]
    
    #plot with lags as x, correlations as y:
    fig = go.Figure(go.Bar(x=lags, y=cross_corr))

    fig.update_yaxes(title_text = "Correlation")
    fig.update_xaxes(title_text = "Lag (days)")

    fig.show()




def main():
    df = load_wx()
    cross_corr = cross_correlation(df)
    plot_correlations(cross_corr)


if __name__ == "__main__":
    main()
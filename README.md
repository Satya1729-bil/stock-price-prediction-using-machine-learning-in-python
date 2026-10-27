# Stock Direction Prediction with an SVM

**Can two simple price features predict whether tomorrow's close will be higher? In this test, no.**

This notebook trains a Support Vector Machine (SVM) to predict whether a stock's next-day closing price will be higher (1) or lower (0) than today's. It uses two features from each day's prices: Open minus Close, and High minus Low. The data is split 80/20 in time order, so the model is tested on later dates than it was trained on.

## Results (from the saved notebook run)

| Model | Test accuracy |
|---|---|
| SVM, RBF kernel (default) | 47.2% |
| SVM, linear kernel | 49.1% |
| SVM, polynomial kernel (degree 3) | 49.0% |
| SVM, sigmoid kernel | 50.5% |

For the default model: ROC AUC 0.47, F1 0.50, training accuracy 53.6%.

Every kernel lands close to 50%, which is what a coin flip would score. Two daily price-range features do not carry enough signal to predict next-day direction. Useful next steps would be richer features (multi-day returns, volume, moving averages) and walk-forward validation.

## Files

- `Stock_Price_Prediction_using_svm_(1).ipynb`: the notebook, with its outputs
- `Tesla.csv`: Tesla daily prices, June 2010 to March 2017

Note: the saved run loads a different daily price file from Google Drive (data starting January 2009), not `Tesla.csv`. To reproduce with the file in this repo, replace the Google Drive download in the second cell with `pd.read_csv("Tesla.csv")`.

## Run it

Open the notebook in Google Colab or Jupyter. It needs `pandas`, `numpy`, `scikit-learn` and `matplotlib`.

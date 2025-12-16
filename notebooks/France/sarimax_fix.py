# Prédiction
preds = fitted.forecast(steps=len(y_test), exog=X_test)
preds.index = y_test.index

# Métriques
# Créer le mask pour MAPE (éviter division par zéro)
mask_mape = y_test > 1.0

mae_sarimax = mean_absolute_error(y_test, preds)
rmse_sarimax = np.sqrt(mean_squared_error(y_test, preds))
r2_sarimax = r2_score(y_test, preds)
mape_sarimax = np.mean(np.abs((y_test[mask_mape] - preds[mask_mape]) / y_test[mask_mape])) * 100

print(f'MAE  : {mae_sarimax:.2f} €/MWh | RMSE : {rmse_sarimax:.2f} €/MWh | R²   : {r2_sarimax:.3f} | MAPE : {mape_sarimax:.2f} %')

# Visualisation
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test.index, y=y_test, name='Réel', line=dict(color='black', width=1)))
fig.add_trace(go.Scatter(x=preds.index, y=preds, name='Prédiction', line=dict(color='#AB63FA', width=2)))
fig.update_layout(title=f'Prédiction VS Réel (MAE: {mae_sarimax:.2f}€)', xaxis_title='Date', yaxis_title='Prix (€/MWh)', height=600, template='plotly_white')
fig.show()

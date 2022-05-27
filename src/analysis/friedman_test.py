from scipy import stats

# seleciona melhores pelo summary
# abrir log_resultados de cada um (array de loss)

mlp1 = []
mlp2 = []
rbf = []
esn = []
elm = []

stat, pvalue = stats.friedmanchisquare(mlp1, mlp2, rbf, esn, elm)

print("Estatística:", stat)
print("P-valor:{:.10f}".format(pvalue))

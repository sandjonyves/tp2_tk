import pandas as pd, numpy as np, pickle, os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings('ignore')

for d in ['models','data','static/plots']: os.makedirs(d, exist_ok=True)

print("="*60)
print("PARTIE 1 : CLASSIFICATION - Census Dataset")
print("="*60)

np.random.seed(42)
n = 1000
age = np.random.randint(18, 90, n)
edu_num = np.random.randint(1, 16, n)
hours = np.random.randint(10, 80, n)
cap_gain = np.random.exponential(500, n)
cap_loss = np.random.exponential(100, n)
education = np.random.choice(['Bachelors','HS-grad','Masters','Doctorate','Some-college'], n)
marital = np.random.choice(['Married-civ-spouse','Divorced','Never-married','Separated','Widowed'], n)
occupation = np.random.choice(['Tech-support','Craft-repair','Sales','Exec-managerial','Prof-specialty'], n)
sex = np.random.choice(['Male','Female'], n)
score_raw = (age-30)*0.03 + edu_num*0.15 + hours*0.03 + cap_gain*0.0002
prob = 1/(1+np.exp(-(score_raw-2.5)))
income = (prob > 0.5).astype(int)

census_df = pd.DataFrame({'age':age,'education':education,'education-num':edu_num,
    'marital-status':marital,'occupation':occupation,'sex':sex,
    'capital-gain':cap_gain,'capital-loss':cap_loss,'hours-per-week':hours,'income':income})
census_df.to_csv('data/census.csv', index=False)
print(f"✓ Census: {census_df.shape}, classes={census_df['income'].value_counts().to_dict()}")

X = pd.get_dummies(census_df.drop('income',axis=1), drop_first=True)
y = census_df['income']
feature_names_census = X.columns.tolist()
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

knn_scores = []
for k in range(1,21):
    knn=KNeighborsClassifier(n_neighbors=k); knn.fit(X_train,y_train)
    knn_scores.append(knn.score(X_test,y_test))
best_k=np.argmax(knn_scores)+1
knn_best=KNeighborsClassifier(n_neighbors=best_k); knn_best.fit(X_train,y_train)

gs_dt=GridSearchCV(DecisionTreeClassifier(random_state=42),{'max_depth':[3,5,7,10],'criterion':['gini','entropy']},cv=5)
gs_dt.fit(X_train,y_train); dt_best=gs_dt.best_estimator_

rf=RandomForestClassifier(n_estimators=100,oob_score=True,random_state=42); rf.fit(X_train,y_train)
gb=GradientBoostingClassifier(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42); gb.fit(X_train,y_train)

scores={'KNN':knn_best.score(X_test,y_test),'Decision Tree':dt_best.score(X_test,y_test),
        'Random Forest':rf.score(X_test,y_test),'Gradient Boosting':gb.score(X_test,y_test)}
best_name=max(scores,key=scores.get)
best_map={'KNN':knn_best,'Decision Tree':dt_best,'Random Forest':rf,'Gradient Boosting':gb}
print(f"Scores: {scores}\n✓ Meilleur: {best_name} ({scores[best_name]:.4f})")

with open('models/census.pkl','wb') as f: pickle.dump(best_map[best_name],f)
with open('models/census_features.pkl','wb') as f: pickle.dump(feature_names_census,f)
with open('models/census_scores.pkl','wb') as f: pickle.dump(scores,f)

fig,ax=plt.subplots(figsize=(10,5))
ax.plot(range(1,21),knn_scores,'o-',color='steelblue',linewidth=2)
ax.axvline(best_k,color='red',linestyle='--',label=f'k optimal={best_k}')
ax.set_xlabel('k'); ax.set_ylabel('Accuracy'); ax.legend(); ax.grid(alpha=0.3)
ax.set_title('Influence du paramètre k - KNN Classification')
plt.tight_layout(); plt.savefig('static/plots/knn_classification.png',dpi=100); plt.close()

fig,ax=plt.subplots(figsize=(8,5))
colors=['#3498db','#e74c3c','#2ecc71','#f39c12']
bars=ax.bar(scores.keys(),scores.values(),color=colors,edgecolor='black')
ax.set_ylim(0.5,1.0); ax.set_ylabel('Accuracy'); ax.set_title('Comparaison des modèles - Classification')
for bar,v in zip(bars,scores.values()):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,f'{v:.3f}',ha='center',fontweight='bold')
ax.grid(axis='y',alpha=0.3); plt.tight_layout()
plt.savefig('static/plots/models_comparison_census.png',dpi=100); plt.close()

fi=pd.Series(rf.feature_importances_,index=feature_names_census).sort_values(ascending=False).head(10)
fig,ax=plt.subplots(figsize=(10,6))
fi.plot(kind='barh',ax=ax,color='steelblue',edgecolor='black')
ax.set_title('Top 10 Variables Importantes - Random Forest'); ax.invert_yaxis()
plt.tight_layout(); plt.savefig('static/plots/feature_importance_census.png',dpi=100); plt.close()

print("\n"+"="*60)
print("PARTIE 2 : RÉGRESSION - Auto-MPG Dataset")
print("="*60)

np.random.seed(42)
n2=392
cyl=np.random.choice([4,6,8],n2,p=[0.5,0.25,0.25])
disp=cyl*28+np.random.normal(0,15,n2)
hp=cyl*18+np.random.normal(0,12,n2)
wt=1500+cyl*300+np.random.normal(0,250,n2)
acc=np.random.normal(15,3,n2)
yr=np.random.randint(70,83,n2)
org=np.random.choice([1,2,3],n2,p=[0.6,0.2,0.2])
mpg=40-0.006*wt-0.1*hp+0.5*acc+0.4*(yr-70)+org+np.random.normal(0,2,n2)

auto_df=pd.DataFrame({'cylinders':cyl,'displacement':disp,'horsepower':hp,'weight':wt,
    'acceleration':acc,'model_year':yr,'origin':org,'mpg':mpg})
auto_df.to_csv('data/auto-mpg.csv',index=False)
print(f"✓ Auto-MPG: {auto_df.shape}, mpg=[{mpg.min():.1f},{mpg.max():.1f}]")

X2=auto_df.drop('mpg',axis=1); y2=auto_df['mpg']
X2_train,X2_test,y2_train,y2_test=train_test_split(X2,y2,test_size=0.2,random_state=42)

scaler=StandardScaler()
X2_tr_sc=scaler.fit_transform(X2_train); X2_te_sc=scaler.transform(X2_test)

knn_r_scores=[]
for k in range(1,21):
    kr=KNeighborsRegressor(n_neighbors=k); kr.fit(X2_tr_sc,y2_train)
    knn_r_scores.append(r2_score(y2_test,kr.predict(X2_te_sc)))
best_k_r=np.argmax(knn_r_scores)+1
knn_r_best=KNeighborsRegressor(n_neighbors=best_k_r); knn_r_best.fit(X2_tr_sc,y2_train)
y_pred_knn=knn_r_best.predict(X2_te_sc)

rf_r=RandomForestRegressor(n_estimators=100,random_state=42); rf_r.fit(X2_tr_sc,y2_train)
y_pred_rf_r=rf_r.predict(X2_te_sc)

gb_r=GradientBoostingRegressor(n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42)
gb_r.fit(X2_tr_sc,y2_train); y_pred_gb_r=gb_r.predict(X2_te_sc)

reg_scores={
    'KNN':{'MAE':mean_absolute_error(y2_test,y_pred_knn),'MSE':mean_squared_error(y2_test,y_pred_knn),'R2':r2_score(y2_test,y_pred_knn)},
    'Random Forest':{'MAE':mean_absolute_error(y2_test,y_pred_rf_r),'MSE':mean_squared_error(y2_test,y_pred_rf_r),'R2':r2_score(y2_test,y_pred_rf_r)},
    'Gradient Boosting':{'MAE':mean_absolute_error(y2_test,y_pred_gb_r),'MSE':mean_squared_error(y2_test,y_pred_gb_r),'R2':r2_score(y2_test,y_pred_gb_r)},
}
best_reg_name=max(reg_scores,key=lambda x:reg_scores[x]['R2'])
best_reg_map={'KNN':knn_r_best,'Random Forest':rf_r,'Gradient Boosting':gb_r}
print(f"R²: { {k:round(v['R2'],4) for k,v in reg_scores.items()} }")
print(f"✓ Meilleur: {best_reg_name} (R²={reg_scores[best_reg_name]['R2']:.4f})")

with open('models/auto-mpg.pkl','wb') as f: pickle.dump(best_reg_map[best_reg_name],f)
with open('models/auto_scaler.pkl','wb') as f: pickle.dump(scaler,f)
with open('models/auto_scores.pkl','wb') as f: pickle.dump(reg_scores,f)

fig,ax=plt.subplots(figsize=(10,5))
ax.plot(range(1,21),knn_r_scores,'o-',color='darkorange',linewidth=2)
ax.axvline(best_k_r,color='red',linestyle='--',label=f'k optimal={best_k_r}')
ax.set_xlabel('k'); ax.set_ylabel('R²'); ax.legend(); ax.grid(alpha=0.3)
ax.set_title('Influence du paramètre k - KNN Régression')
plt.tight_layout(); plt.savefig('static/plots/knn_regression.png',dpi=100); plt.close()

r2_vals={k:v['R2'] for k,v in reg_scores.items()}
fig,ax=plt.subplots(figsize=(8,5))
bars2=ax.bar(r2_vals.keys(),r2_vals.values(),color=['#3498db','#2ecc71','#f39c12'],edgecolor='black')
ax.set_ylim(0,1.05); ax.set_ylabel('R²'); ax.set_title('Comparaison des modèles - Régression Auto-MPG')
for bar,v in zip(bars2,r2_vals.values()):
    ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,f'{v:.3f}',ha='center',fontweight='bold')
ax.grid(axis='y',alpha=0.3); plt.tight_layout()
plt.savefig('static/plots/models_comparison_regression.png',dpi=100); plt.close()

best_preds={'KNN':y_pred_knn,'Random Forest':y_pred_rf_r,'Gradient Boosting':y_pred_gb_r}[best_reg_name]
fig,ax=plt.subplots(figsize=(7,7))
ax.scatter(y2_test,best_preds,alpha=0.6,color='steelblue',edgecolors='black',linewidth=0.3)
mn,mx=min(y2_test.min(),best_preds.min()),max(y2_test.max(),best_preds.max())
ax.plot([mn,mx],[mn,mx],'r--',linewidth=2,label='Parfait')
ax.set_xlabel('Valeurs réelles (mpg)'); ax.set_ylabel('Valeurs prédites (mpg)')
ax.set_title(f'Prédictions vs Réelles - {best_reg_name}'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('static/plots/regression_scatter.png',dpi=100); plt.close()

print("\n✓ Tous les modèles générés avec succès!")

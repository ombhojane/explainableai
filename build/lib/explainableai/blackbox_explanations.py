# blackbox_explanations.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.model_selection import cross_val_score
from alibi.explainers import AnchorTabular, CounterfactualProto
from alibi.explainers.ale import ALE

class BlackBoxExplainer:
    def __init__(self, model, X, y, feature_names):
        self.model = model
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.is_classifier = hasattr(model, "predict_proba")

    def explain(self):
        results = {}
        results['anchor'] = self.anchor_explanation()
        results['counterfactual'] = self.counterfactual_explanation()
        results['ale'] = self.ale_explanation()
        results['permutation_importance'] = self.permutation_importance()
        return results

    def anchor_explanation(self):
        explainer = AnchorTabular(self.model.predict, feature_names=self.feature_names)
        explainer.fit(self.X)
        
        # Explain a random instance
        i = np.random.randint(0, self.X.shape[0])
        explanation = explainer.explain(self.X[i])
        
        return {
            'anchor': explanation.anchor,
            'precision': explanation.precision,
            'coverage': explanation.coverage
        }

    def counterfactual_explanation(self):
        cf_explainer = CounterfactualProto(self.model.predict, shape=self.X.shape[1], feature_names=self.feature_names)
        cf_explainer.fit(self.X)
        
        # Explain a random instance
        i = np.random.randint(0, self.X.shape[0])
        explanation = cf_explainer.explain(self.X[i])
        
        return {
            'counterfactual': explanation.cf['X'],
            'distance': explanation.cf['distance']
        }

    def ale_explanation(self):
        ale = ALE(self.model.predict, feature_names=self.feature_names)
        exp = ale.explain(self.X)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        exp.plot(features=[0, 1], ax=axes)
        plt.tight_layout()
        plt.savefig('ale_explanation.png')
        plt.close()
        
        return "ALE Explanation saved as 'ale_explanation.png'"

    def permutation_importance(self):
        from sklearn.inspection import permutation_importance
        
        r = permutation_importance(self.model, self.X, self.y, n_repeats=10, random_state=0)
        
        importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': r.importances_mean,
            'std': r.importances_std
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.bar(importance['feature'], importance['importance'])
        plt.xticks(rotation=90)
        plt.title('Permutation Importance')
        plt.tight_layout()
        plt.savefig('permutation_importance.png')
        plt.close()
        
        return importance.to_dict(orient='records')
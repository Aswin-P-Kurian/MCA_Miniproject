from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

class DisasterAidPredictor:
    def __init__(self):
        self.models = {}
        self.encoders = {}
        self.feature_columns = []
        self.load_available_models() # Corrected method name

    def load_available_models(self): # Corrected method name
        """Load available models with NEW filenames"""
        try:
            # Load classification models (NEW FILENAMES)
            with open('models/clf_priority_level_model.pkl', 'rb') as f:
                self.models['priority'] = pickle.load(f)
            with open('models/clf_urgent_need_model.pkl', 'rb') as f:
                self.models['need'] = pickle.load(f)
            with open('models/clf_suggested_action_model.pkl', 'rb') as f:
                self.models['action'] = pickle.load(f)

            # Load regression models (NEW FILENAMES)
            with open('models/reg_response_urgency_model.pkl', 'rb') as f:
                self.models['response'] = pickle.load(f)
            with open('models/reg_economic_loss_model.pkl', 'rb') as f:
                self.models['loss'] = pickle.load(f)
            with open('models/reg_medical_team_model.pkl', 'rb') as f:
                self.models['medical'] = pickle.load(f)
            with open('models/reg_rescue_team_model.pkl', 'rb') as f:
                self.models['rescue'] = pickle.load(f)

            # Load label encoders (NEW FILENAMES)
            with open('models/le_priority_level.pkl', 'rb') as f:
                self.encoders['priority'] = pickle.load(f)
            with open('models/le_urgent_need.pkl', 'rb') as f:
                self.encoders['need'] = pickle.load(f)
            with open('models/le_suggested_action.pkl', 'rb') as f:
                self.encoders['action'] = pickle.load(f)

            # Load feature columns (ASSUMING FILENAME REMAINS 'features.pkl')
            # If you renamed this file too, update the path below
            with open('models/features.pkl', 'rb') as f:
                self.feature_columns = pickle.load(f)

            print("✅ All models and encoders loaded successfully!")
            print(f"📊 Expected features: {len(self.feature_columns)}")

        except Exception as e:
            print(f"❌ Error loading models: {e}")
            # Initialize with None values as fallback
            for model_name in ['priority', 'need', 'action', 'response', 'loss', 'medical', 'rescue']:
                self.models[model_name] = None
            for encoder_name in ['priority', 'need', 'action']:
                self.encoders[encoder_name] = None
            self.feature_columns = [] # Ensure feature_columns is empty list on failure

    def create_features_for_prediction(self, area_data):
        """Create features that match the training data structure"""
        # Start with all zeros for all expected features
        if not self.feature_columns:
            print("⚠️ Feature columns not loaded. Cannot create features.")
            return {} # Return empty dict if features aren't loaded

        features = {col: 0 for col in self.feature_columns}

        # Map the basic features we have from the form
        # Ensure keys match those sent from your JavaScript 'getFormDataForApi'
        feature_mapping = {
            'total_population': 'Total Population',
            'population_density': 'Population Density',
            'elderly_percentage': 'Percentage of Elderly People (65+)',
            'children_percentage': 'Percentage of Children (0–12)',
            'distance_to_hospital': 'Distance to Nearest Hospital (km)',
            'distance_to_city': 'Distance to Nearest City (km)',
            'severity': 'Severity of Disaster (Scale 1–5)',
            'estimated_injuries': 'Number of Injuries (Estimated)',
            'displaced': 'Number of Homeless or Displaced',
            'area_affected': 'Area of Region Affected (in sq. km)',
            'infrastructure_damage': '% of Area Infrastructure Damaged', # Corrected name
            'medical_stock': 'Current Stock of Medical Supplies', # Corrected name
            'food_stock': 'Current Stock of Food Supplies', # Corrected name
            'volunteers': 'Number of Active Volunteers in Area', # Corrected name
            'water_demand': 'Water Demand Estimate (litres per day)', # Corrected name
            'food_requirement': 'Food Supply Requirement (kg/day)', # Corrected name
            'time_since_disaster': 'Estimated Time Since Disaster (in hours)', # Corrected name
            'medical_facility_score': 'Medical Facility Availability Score (0–5)', # Corrected name
            'road_access_score': 'Road Accessibility (Rating 1–5)', # Corrected name
            'past_disasters_frequency': 'Frequency of Past Disasters in Area', # Corrected name
            'economic_loss_estimate': 'Economic_Loss_Estimate', # Need input or default?
            'medical_teams_required': 'Medical_Team_Required', # Need input or default?
            'rescue_teams_required': 'Rescue_Team_Required' # Need input or default?
        }

        # Fill in the numerical features we have data for
        for input_key, feature_name in feature_mapping.items():
            if feature_name in features:
                # Use .get with a default of 0 to handle potentially missing keys
                features[feature_name] = float(area_data.get(input_key, 0))

        # Handle categorical variables - map input values to one-hot encoded feature names
        categorical_fields_mapping = {
            'disaster_type': 'Type of Disaster',
            'urban_rural': 'Urban / Rural Classification',
            'weather_conditions': 'Weather Condition Status',
            'vulnerable_groups': 'Presence of Vulnerable Groups',
            'internet_available': 'Internet Access Availability',
            'electricity_available': 'Electricity Availability',
            'recent_aid_provided': 'Recent Aid Provided? (Yes/No)', # Added from input form
            # Add other categorical fields your model expects if they exist in the input
            'casualty_risk_level': 'Casualty_Risk_Level' # Example if needed
        }

        for input_key, base_feature_name in categorical_fields_mapping.items():
            value = area_data.get(input_key)
            if value: # Check if the value exists and is not empty
                # Construct the one-hot encoded feature name EXACTLY as it appeared during training
                # Example: 'Type of Disaster_Flood' or 'Urban / Rural Classification_Urban'
                # You might need to adjust this based on how your 'features.pkl' names are stored
                # This assumes scikit-learn's default get_dummies style naming or OneHotEncoder naming
                one_hot_feature_name = f"{base_feature_name}_{value}"
                if one_hot_feature_name in features:
                    features[one_hot_feature_name] = 1
                else:
                    # If the exact feature name isn't found, try a simple title case version
                    # (Less robust, depends heavily on consistent naming in features.pkl)
                    simple_feature_name = f"{base_feature_name.title()} = {value}"
                    if simple_feature_name in features:
                         features[simple_feature_name] = 1
                    else:
                         print(f"⚠️ Warning: Could not find matching feature for {input_key}='{value}'. Tried '{one_hot_feature_name}' and '{simple_feature_name}'.")


        return features

    def predict_all(self, areas_data):
        """Run predictions for all areas"""
        results = {
            'areas': [],
            'resources': {
                'rescue_total': 0,
                'medical_total': 0,
                'food_total': 0, # These might need better calculation
                'water_total': 0 # These might need better calculation
            },
            'priorities': {'High': 0, 'Medium': 0, 'Low': 0}
        }

        for i, area_data in enumerate(areas_data):
            try:
                print(f"🔍 Predicting for: {area_data.get('area_name', f'Area {i+1}')}")

                # Create features dictionary
                features_dict = self.create_features_for_prediction(area_data)

                # Convert dictionary to DataFrame with correct column order
                features_df = pd.DataFrame([features_dict])[self.feature_columns]

                # --- Get ML predictions ---
                priority_pred = self._predict_classification('priority', features_df, area_data)
                need_pred = self._predict_classification('need', features_df, area_data)
                action_pred = self._predict_classification('action', features_df, area_data)

                response_pred = self._predict_regression('response', features_df, area_data)
                loss_pred = self._predict_regression('loss', features_df, area_data)
                medical_pred = self._predict_regression('medical', features_df, area_data)
                rescue_pred = self._predict_regression('rescue', features_df, area_data)

                # --- Calculate derived values ---
                urgency, urgency_hours = self._calculate_urgency(response_pred, priority_pred) # Use predicted response time
                food_supplies = self._calculate_food_supplies(area_data, priority_pred)
                water_supply = self._calculate_water_supply(area_data, priority_pred)

                area_result = {
                    'area': area_data.get('area_name', f'Area {i+1}'),
                    'priority': priority_pred,
                    'need_level': need_pred, # Assuming 'need' model predicts this
                    'action': action_pred,
                    'urgency': urgency,
                    'urgency_hours': urgency_hours,
                    'population': int(area_data.get('total_population', 0)),
                    'rescue_teams': int(rescue_pred), # Use ML prediction
                    'medical_teams': int(medical_pred), # Use ML prediction
                    'food_supplies': food_supplies, # Calculated %
                    'water_supply': water_supply, # Calculated %
                    'estimated_loss': float(loss_pred) # Use ML prediction
                }

                results['areas'].append(area_result)
                # Increment priority count safely
                if priority_pred in results['priorities']:
                    results['priorities'][priority_pred] += 1
                else:
                     results['priorities'][priority_pred] = 1 # Handle unexpected priority levels


                # Aggregate resources based on ML predictions
                results['resources']['rescue_total'] += int(rescue_pred)
                results['resources']['medical_total'] += int(medical_pred)
                # Base food/water might need refinement based on need_pred or population
                results['resources']['food_total'] += 100
                results['resources']['water_total'] += 100

                print(f"✅ {area_result['area']}: Priority={priority_pred}, Need={need_pred}, Action={action_pred}")

            except Exception as e:
                print(f"❌ Prediction Error for area {i+1}: {e}")
                # Append a fallback prediction if processing fails for an area
                fallback_pred = self._get_fallback_prediction(area_data, i + 1)
                results['areas'].append(fallback_pred)
                if fallback_pred['priority'] in results['priorities']:
                     results['priorities'][fallback_pred['priority']] += 1

        print(f"📊 Final results count: {results['priorities']}")
        return results

    def _predict_classification(self, model_type, features_df, area_data):
        """Helper for classification models with fallback"""
        if self.models.get(model_type) and self.encoders.get(model_type):
            try:
                pred_encoded = self.models[model_type].predict(features_df)[0]
                return self.encoders[model_type].inverse_transform([pred_encoded])[0]
            except Exception as e:
                print(f"⚠️ ML prediction failed for {model_type}: {e}. Using fallback.")

        # Fallback Logic based on severity
        severity = area_data.get('severity', 3)
        if model_type == 'priority':
            return "High" if severity >= 4 else "Medium" if severity >= 2 else "Low"
        elif model_type == 'need':
            # Example fallback: Map severity/priority to need
             priority = self._predict_classification('priority', features_df, area_data) # Get priority fallback first
             if priority == 'High': return 'Rescue'
             if priority == 'Medium': return 'Medical'
             return 'Food' # Default for Low
        elif model_type == 'action':
             priority = self._predict_classification('priority', features_df, area_data) # Get priority fallback first
             if priority == 'High': return "Deploy rescue and medical teams immediately."
             if priority == 'Medium': return "Assess damage and provide essential supplies."
             return "Monitor situation and provide support as needed."
        return f"Unknown_{model_type}" # Default fallback

    def _predict_regression(self, model_type, features_df, area_data):
         """Helper for regression models with fallback"""
         if self.models.get(model_type):
             try:
                 # Ensure prediction is a float and handle potential NaNs
                 prediction = float(self.models[model_type].predict(features_df)[0])
                 return prediction if np.isfinite(prediction) else 0.0 # Return 0 if NaN/Inf
             except Exception as e:
                 print(f"⚠️ ML prediction failed for {model_type}: {e}. Using fallback.")

         # Fallback Logic (Examples - refine these)
         severity = area_data.get('severity', 3)
         population = area_data.get('total_population', 1000)
         injuries = area_data.get('estimated_injuries', 0)
         displaced = area_data.get('displaced', 0)
         infra_damage = area_data.get('infrastructure_damage', 0)

         if model_type == 'response':
             return max(1.0, 24.0 / severity) # Simple inverse severity relation
         elif model_type == 'loss':
             return float(infra_damage * population / 1000.0) # Based on damage and pop
         elif model_type == 'medical':
             return max(1.0, injuries / 50.0) # Teams per 50 injuries
         elif model_type == 'rescue':
             return max(1.0, displaced / 200.0) # Teams per 200 displaced
         return 0.0 # Default fallback

    def _calculate_urgency(self, response_time_pred, priority_pred):
        """Convert predicted response time to urgency level"""
        # More nuanced urgency based on priority AND predicted time
        if priority_pred == "High" or response_time_pred <= 6:
            level = "Critical"
            hours = max(1, int(response_time_pred / 2)) # Critical means faster than predicted
        elif priority_pred == "Medium" or response_time_pred <= 24:
            level = "Moderate"
            hours = max(1, int(response_time_pred))
        else:
            level = "Low"
            hours = max(1, int(response_time_pred * 1.5)) # Low priority might take longer
        return level, hours

    # --- Calculation helpers (Can keep these simpler or enhance further) ---
    def _calculate_food_supplies(self, area_data, priority):
        severity = area_data.get('severity', 3)
        base = max(20, 100 - (severity * 15)) # More aggressive reduction
        return min(100, int(base))

    def _calculate_water_supply(self, area_data, priority):
        severity = area_data.get('severity', 3)
        base = max(30, 100 - (severity * 12)) # More aggressive reduction
        return min(100, int(base))

    def _calculate_estimated_loss(self, area_data, priority):
        # This is now predicted by reg_loss model, use that value
        # This function could be removed or used only in fallback
        severity = area_data.get('severity', 3)
        infra_damage = area_data.get('infrastructure_damage', 0)
        return float(infra_damage * (severity / 5.0) * 1.2) # Simple fallback if needed

    def _get_fallback_prediction(self, area_data, index):
        """Provide fallback predictions if models fail"""
        severity = area_data.get('severity', 3)
        priority = "High" if severity >= 4 else "Medium" if severity >= 2 else "Low"
        urgency, urgency_hours = self._calculate_urgency({}, priority) # Use default urgency calc

        return {
            'area': area_data.get('area_name', f'Area {index}'),
            'priority': priority,
            'need_level': "Critical" if severity >= 4 else "High",
            'action': "Fallback: Assess situation and deploy resources based on severity.",
            'urgency': urgency,
            'urgency_hours': urgency_hours,
            'population': area_data.get('total_population', 0),
            'rescue_teams': self._calculate_rescue_teams(area_data, priority),
            'medical_teams': self._calculate_medical_teams(area_data, priority),
            'food_supplies': self._calculate_food_supplies(area_data, priority),
            'water_supply': self._calculate_water_supply(area_data, priority),
            'estimated_loss': self._calculate_estimated_loss(area_data, priority)
        }

# Initialize predictor
predictor = DisasterAidPredictor()

@app.route('/')
def landing():
    return render_template('index.html')

@app.route('/input')
def index():
    return render_template('input.html')

@app.route('/submit', methods=['POST'])
def submit_data():
    try:
        data = request.get_json()
        if not data or 'areas' not in data:
            return jsonify({'error': 'No area data received'}), 400

        areas_data = data['areas']
        print(f"📨 Received data for {len(areas_data)} areas...")

        # --- VALIDATION ---
        required_fields = ['area_name', 'total_population', 'disaster_type', 'severity']
        for i, area in enumerate(areas_data):
            for field in required_fields:
                 # Check if field exists and is not empty or zero for numbers
                 value = area.get(field)
                 is_missing_or_invalid = (
                     value is None or
                     (isinstance(value, str) and not value.strip()) or
                     (isinstance(value, (int, float)) and value <= 0 and field == 'total_population') # Pop must be > 0
                 )
                 if is_missing_or_invalid:
                      error_msg = f"Area {i+1} ('{area.get('area_name', 'Unnamed')}') missing or has invalid required field: '{field}'"
                      print(f"❌ Validation Error: {error_msg}")
                      return jsonify({'error': error_msg}), 400


        # --- PREDICTION ---
        predictions = predictor.predict_all(areas_data)
        predictions['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        session['prediction_results'] = predictions

        return jsonify({
            'success': True,
            'message': f'Successfully analyzed {len(areas_data)} areas.',
            'redirect_url': url_for('results') # Use url_for for robustness
        })

    except json.JSONDecodeError:
        print("❌ Submission Error: Invalid JSON received.")
        return jsonify({'error': 'Invalid JSON data received.'}), 400
    except KeyError as e:
         print(f"❌ Submission Error: Missing key {e} in input data.")
         return jsonify({'error': f'Missing expected data field: {e}'}), 400
    except Exception as e:
        print(f"❌ Submission error: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return jsonify({'error': f'An unexpected error occurred during processing: {str(e)}'}), 500


@app.route('/results')
def results():
    results_data = session.get('prediction_results')
    if not results_data:
        print("⚠️ No results found in session, redirecting to input.")
        return redirect(url_for('index')) # Redirect to input page if no results

    # Format data specifically for the result.html template's expectations
    formatted_results = format_results_for_template(results_data)
    #print(f"📊 Sending formatted results to template: {json.dumps(formatted_results, indent=2)}") # Debug print
    return render_template('result.html', modelOutput=formatted_results)

def format_results_for_template(predictions):
    """Convert prediction results to match result.html expected format"""
    # Recalculate priority counts directly from the final predictions['areas'] list
    priority_counts = {'High': 0, 'Medium': 0, 'Low': 0}
    for area in predictions.get('areas', []):
        p = area.get('priority')
        if p in priority_counts:
            priority_counts[p] += 1

    return {
        'timestamp': predictions.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        'priorities': priority_counts,
        'areas': predictions.get('areas', []),
        'resources': predictions.get('resources', {})
    }

if __name__ == '__main__':
    # Ensure the 'models' directory exists relative to app.py
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    if not os.path.exists(models_dir):
        print(f"⚠️ Warning: 'models' directory not found at {models_dir}")
    app.run(debug=True, host='0.0.0.0', port=5000) 
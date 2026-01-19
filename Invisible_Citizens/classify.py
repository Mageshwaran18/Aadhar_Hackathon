import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CLASSIFICATION DICTIONARIES
# ============================================================================

TRIBAL_DOMINATED_DISTRICTS = {
    'assam': [
        'chirang', 'kokrajhar', 'dhemaji', 'karbi_anglong', 'dima_hasao',
        'goalpara', 'barpeta', 'sonitpur', 'lakhimpur'
    ],
    'bihar': ['purnia'],
    'chhattisgarh': [
        'bijapur', 'dakshin_bastar_dantewada', 'bastar', 'kondagaon', 
        'sukma', 'narayanpur', 'kanker', 'gariaband', 'korea', 
        'surguja', 'jashpur', 'raigarh'
    ],
    'gujarat': [
        'dahod', 'arvalli', 'panch_mahals', 'chhotaudepur', 'valsad',
        'narmada', 'tapi', 'dang'
    ],
    'jharkhand': [
        'ranchi', 'garhwa', 'pakur', 'gumla', 'simdega', 'khunti',
        'west_singhbhum', 'east_singhbhum', 'saraikela_kharsawan',
        'lohardaga', 'dumka', 'jamtara', 'sahebganj'
    ],
    'madhya_pradesh': [
        'jhabua', 'alirajpur', 'barwani', 'burhanpur', 'dhar',
        'mandla', 'dindori', 'umaria', 'anuppur', 'shahdol',
        'sidhi', 'singrauli', 'betul', 'chhindwara'
    ],
    'maharashtra': [
        'nandurbar', 'dhule', 'palghar', 'nashik', 'thane',
        'gadchiroli', 'chandrapur', 'yavatmal', 'amravati'
    ],
    'meghalaya': [
        'south_west_garo_hills', 'west_garo_hills', 'west_jaintia_hills',
        'east_khasi_hills', 'west_khasi_hills', 'south_west_khasi_hills',
        'east_garo_hills', 'north_garo_hills', 'south_garo_hills',
        'ri_bhoi', 'east_jaintia_hills'
    ],
    'mizoram': [
        'lawngtlai', 'mamit', 'kolasib', 'aizawl', 'champhai',
        'serchhip', 'lunglei', 'saiha'
    ],
    'odisha': [
        'mayurbhanj', 'keonjhar', 'sundargarh', 'koraput', 'malkangiri',
        'rayagada', 'nabarangpur', 'kalahandi', 'kandhamal', 'gajapati'
    ],
    'rajasthan': ['udaipur', 'dungarpur', 'banswara', 'pratapgarh', 'sirohi'],
    'telangana': [
        'adilabad', 'komaram_bheem_asifabad', 'mancherial', 
        'bhadradri_kothagudem', 'khammam', 'mahabubabad'
    ],
    'west_bengal': [
        'jalpaiguri', 'alipurduar', 'darjeeling', 'purulia',
        'bankura', 'jhargram', 'paschim_medinipur'
    ]
}

FOREST_HILLY_DISTRICTS = {
    'assam': [
        'golaghat', 'kamrup', 'cachar', 'hailakandi', 'marigaon',
        'dhubri', 'hojai'
    ],
    'jammu_and_kashmir': [
        'doda', 'poonch', 'kishtwar', 'ramban', 'rajouri', 'reasi',
        'kathua', 'udhampur', 'samba', 'jammu', 'anantnag', 'kulgam',
        'pulwama', 'shopian', 'budgam', 'baramulla', 'bandipora',
        'ganderbal', 'kupwara', 'srinagar'
    ],
    'himachal_pradesh': [
        'shimla', 'kinnaur', 'kullu', 'mandi', 'chamba', 'kangra',
        'una', 'hamirpur', 'bilaspur', 'solan', 'sirmaur', 'lahaul_spiti'
    ],
    'uttarakhand': [
        'almora', 'bageshwar', 'chamoli', 'champawat', 'dehradun',
        'haridwar', 'nainital', 'pauri_garhwal', 'pithoragarh',
        'rudraprayag', 'tehri_garhwal', 'udham_singh_nagar', 'uttarkashi'
    ],
    'karnataka': [
        'uttara_kannada', 'udupi', 'dakshina_kannada', 'chikmagalur',
        'hassan', 'kodagu', 'shimoga', 'chickballapur'
    ],
    'kerala': [
        'idukki', 'wayanad', 'palakkad', 'malappuram', 'kozhikode',
        'kannur', 'kasaragod', 'pathanamthitta', 'kottayam'
    ],
    'madhya_pradesh': ['panna', 'katni', 'sehore'],
    'odisha': ['jagatsinghapur', 'angul', 'dhenkanal', 'sambalpur']
}

LOW_LITERACY_DISTRICTS = {
    'bihar': [
        'sitamarhi', 'purnia', 'saharsa', 'madhepura', 'araria',
        'kishanganj', 'katihar', 'purnea', 'supaul', 'sheohar'
    ],
    'uttar_pradesh': [
        'bahraich', 'bareilly', 'sitapur', 'etah', 'siddharthnagar',
        'shravasti', 'balrampur', 'rampur', 'budaun'
    ],
    'rajasthan': [
        'jalore', 'sirohi', 'karauli', 'dhaulpur', 'pratapgarh',
        'banswara', 'dungarpur', 'sawai_madhopur'
    ],
    'madhya_pradesh': [
        'alirajpur', 'jhabua', 'barwani', 'sheopur', 'ashoknagar',
        'singrauli', 'bhind', 'morena'
    ],
    'jharkhand': ['pakur', 'sahebganj', 'godda', 'dumka', 'jamtara'],
    'chhattisgarh': [
        'bijapur', 'dakshin_bastar_dantewada', 'sukma', 'narayanpur'
    ],
    'assam': ['dhubri', 'barpeta', 'goalpara', 'chirang', 'dhemaji'],
    'odisha': ['nabarangpur', 'malkangiri', 'rayagada', 'kalahandi'],
    'west_bengal': [
        'uttar_dinajpur', 'dakshin_dinajpur', 'murshidabad', 'malda'
    ]
}

MIGRATION_SOURCE_DISTRICTS = {
    'bihar': [
        'sitamarhi', 'purnia', 'madhubani', 'darbhanga', 'samastipur',
        'muzaffarpur', 'gopalganj', 'siwan', 'saran', 'vaishali'
    ],
    'uttar_pradesh': [
        'bahraich', 'sitapur', 'bareilly', 'etah', 'siddharthnagar',
        'azamgarh', 'mau', 'ballia', 'deoria', 'gorakhpur', 'basti'
    ],
    'jharkhand': ['ranchi', 'garhwa', 'pakur', 'palamu', 'gumla', 'latehar'],
    'west_bengal': [
        'murshidabad', 'malda', 'uttar_dinajpur', 'dakshin_dinajpur',
        'birbhum', 'purba_bardhaman'
    ],
    'odisha': ['ganjam', 'balangir', 'kalahandi', 'nuapada', 'bargarh'],
    'rajasthan': ['nagaur', 'alwar', 'bharatpur', 'sawai_madhopur'],
    'madhya_pradesh': [
        'satna', 'rewa', 'sidhi', 'singrauli', 'shahdol', 'umaria'
    ]
}

MIGRATION_DESTINATION_DISTRICTS = {
    'delhi': [
        'east', 'north', 'south', 'west', 'central', 'north_west',
        'north_east', 'south_west', 'south_east', 'new_delhi', 'shahdara'
    ],
    'maharashtra': [
        'mumbai', 'mumbai_suburban', 'thane', 'pune', 'pimpri_chinchwad',
        'nagpur', 'nashik', 'aurangabad'
    ],
    'karnataka': [
        'bengaluru_urban', 'bengaluru_rural', 'bengaluru_south',
        'mysuru', 'mangaluru'
    ],
    'gujarat': [
        'ahmedabad', 'surat', 'vadodara', 'rajkot', 'gandhinagar',
        'bhavnagar', 'jamnagar'
    ],
    'tamil_nadu': [
        'chennai', 'coimbatore', 'madurai', 'tiruchirappalli', 'salem',
        'tiruppur', 'erode'
    ],
    'haryana': [
        'gurgaon', 'faridabad', 'ghaziabad', 'noida', 'greater_noida',
        'sonipat', 'panipat', 'rohtak'
    ],
    'punjab': [
        'ludhiana', 'amritsar', 'jalandhar', 'patiala', 'bathinda', 'kapurthala'
    ],
    'uttar_pradesh': [
        'agra', 'lucknow', 'kanpur', 'meerut', 'ghaziabad', 'noida'
    ]
}

REMOTE_RURAL_DISTRICTS = {
    'jammu_and_kashmir': ['kishtwar', 'doda', 'ramban', 'poonch', 'rajouri'],
    'himachal_pradesh': ['kinnaur', 'lahaul_spiti', 'chamba'],
    'uttarakhand': ['uttarkashi', 'chamoli', 'pithoragarh', 'rudraprayag'],
    'rajasthan': [
        'jaisalmer', 'barmer', 'bikaner', 'karauli', 'sirohi'
    ],
    'gujarat': [
        'kachchh', 'patan', 'banas_kantha', 'gir_somnath',
        'devbhumi_dwarka', 'morbi', 'botad', 'porbandar'
    ],
    'madhya_pradesh': [
        'sheopur', 'ashoknagar', 'guna', 'chhatarpur', 'panna'
    ],
    'chhattisgarh': [
        'bijapur', 'dakshin_bastar_dantewada', 'sukma', 'narayanpur'
    ],
    'assam': ['dhemaji', 'hojai'],
    'arunachal_pradesh': [
        'anjaw', 'changlang', 'dibang_valley', 'east_kameng',
        'east_siang', 'kurung_kumey', 'lohit', 'lower_dibang_valley',
        'lower_subansiri', 'papum_pare', 'tawang', 'tirap',
        'upper_siang', 'upper_subansiri', 'west_kameng', 'west_siang'
    ]
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def normalize_name(name):
    """Normalize state/district names for matching"""
    if pd.isna(name):
        return ''
    return str(name).lower().strip().replace(' ', '_').replace('-', '_')

def classify_district(state, district):
    """Classify a district based on all characteristics"""
    state_norm = normalize_name(state)
    district_norm = normalize_name(district)
    
    classification = {
        'is_tribal': False,
        'is_forest_hilly': False,
        'is_low_literacy': False,
        'is_migration_source': False,
        'is_migration_destination': False,
        'is_remote_rural': False
    }
    
    # Check each classification
    if state_norm in TRIBAL_DOMINATED_DISTRICTS:
        if district_norm in TRIBAL_DOMINATED_DISTRICTS[state_norm]:
            classification['is_tribal'] = True
    
    if state_norm in FOREST_HILLY_DISTRICTS:
        if district_norm in FOREST_HILLY_DISTRICTS[state_norm]:
            classification['is_forest_hilly'] = True
    
    if state_norm in LOW_LITERACY_DISTRICTS:
        if district_norm in LOW_LITERACY_DISTRICTS[state_norm]:
            classification['is_low_literacy'] = True
    
    if state_norm in MIGRATION_SOURCE_DISTRICTS:
        if district_norm in MIGRATION_SOURCE_DISTRICTS[state_norm]:
            classification['is_migration_source'] = True
    
    if state_norm in MIGRATION_DESTINATION_DISTRICTS:
        if district_norm in MIGRATION_DESTINATION_DISTRICTS[state_norm]:
            classification['is_migration_destination'] = True
    
    if state_norm in REMOTE_RURAL_DISTRICTS:
        if district_norm in REMOTE_RURAL_DISTRICTS[state_norm]:
            classification['is_remote_rural'] = True
    
    return classification

# ============================================================================
# MAIN ENRICHMENT FUNCTION
# ============================================================================

def enrich_high_risk_pincodes(input_path, output_path=None):
    """
    Read high-risk pincodes CSV and enrich with classifications
    
    Parameters:
    -----------
    input_path : str
        Path to high_risk_pincodes.csv
    output_path : str, optional
        Path to save enriched CSV. If None, creates default path.
    
    Returns:
    --------
    enriched_df : DataFrame
        Enriched dataframe with all classifications
    """
    
    print("="*80)
    print("ENRICHING HIGH-RISK PINCODES WITH CLASSIFICATIONS")
    print("="*80)
    
    # Load data
    print(f"\nLoading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"✓ Loaded {len(df)} high-risk pincodes")
    print(f"Columns: {df.columns.tolist()}")
    
    # Apply classifications
    print("\nApplying district classifications...")
    
    classifications = df.apply(
        lambda row: classify_district(row['state'], row['district']), 
        axis=1
    )
    
    # Convert list of dicts to DataFrame
    classification_df = pd.DataFrame(classifications.tolist())
    
    # Add classification columns to original dataframe
    enriched_df = pd.concat([df, classification_df], axis=1)
    
    # ========================================================================
    # LAYER A: SETTLEMENT TYPE
    # ========================================================================
    print("\nLayer A: Classifying Settlement Type...")
    
    def get_settlement_type(row):
        """Determine settlement type based on classifications"""
        
        # Migration destinations are typically urban
        if row['is_migration_destination']:
            return 'Urban'
        
        # Remote rural takes priority
        if row['is_remote_rural']:
            return 'Remote Rural'
        
        # Tribal/forest/hilly areas are typically rural
        if row['is_tribal'] or row['is_forest_hilly']:
            return 'Rural'
        
        # Default classification
        # You could add population density logic here if you have the data
        return 'Rural'
    
    enriched_df['Settlement_Type'] = enriched_df.apply(get_settlement_type, axis=1)
    
    settlement_dist = enriched_df['Settlement_Type'].value_counts()
    print(f"Settlement Type Distribution:\n{settlement_dist}\n")
    
    # ========================================================================
    # LAYER B: TRIBAL/FOREST TYPE
    # ========================================================================
    print("Layer B: Classifying Tribal/Forest Type...")
    
    def get_tribal_forest_type(row):
        """Determine tribal/forest classification"""
        if row['is_tribal']:
            return 'Tribal-dominated'
        elif row['is_forest_hilly']:
            return 'Forest / Hilly'
        else:
            return 'Normal Rural'
    
    enriched_df['Tribal_Forest_Type'] = enriched_df.apply(get_tribal_forest_type, axis=1)
    
    tribal_dist = enriched_df['Tribal_Forest_Type'].value_counts()
    print(f"Tribal/Forest Type Distribution:\n{tribal_dist}\n")
    
    # ========================================================================
    # LAYER C: LITERACY CATEGORY
    # ========================================================================
    print("Layer C: Classifying Literacy Level...")
    
    def get_literacy_category(row):
        """Determine literacy category"""
        if row['is_low_literacy']:
            return 'Low Literacy'
        else:
            # Without actual literacy rate data, we use medium as default
            return 'Medium Literacy'
    
    enriched_df['Literacy_Category'] = enriched_df.apply(get_literacy_category, axis=1)
    
    literacy_dist = enriched_df['Literacy_Category'].value_counts()
    print(f"Literacy Category Distribution:\n{literacy_dist}\n")
    
    # ========================================================================
    # LAYER D: MIGRATION CATEGORY
    # ========================================================================
    print("Layer D: Classifying Migration Pattern...")
    
    def get_migration_category(row):
        """Determine migration pattern"""
        if row['is_migration_source']:
            return 'High Out-Migration'
        elif row['is_migration_destination']:
            return 'High In-Migration'
        else:
            return 'Low Migration'
    
    enriched_df['Migration_Category'] = enriched_df.apply(get_migration_category, axis=1)
    
    migration_dist = enriched_df['Migration_Category'].value_counts()
    print(f"Migration Category Distribution:\n{migration_dist}\n")
    
    # ========================================================================
    # CREATE COMPREHENSIVE RISK PROFILE
    # ========================================================================
    print("Creating comprehensive risk profiles...")
    
    def create_risk_profile(row):
        """Create human-readable risk profile"""
        profile_parts = []
        
        # Settlement
        profile_parts.append(row['Settlement_Type'])
        
        # Tribal/Forest
        if row['Tribal_Forest_Type'] != 'Normal Rural':
            profile_parts.append(row['Tribal_Forest_Type'])
        
        # Literacy
        if row['Literacy_Category'] == 'Low Literacy':
            profile_parts.append('Low Literacy')
        
        # Migration
        if 'High' in row['Migration_Category']:
            profile_parts.append(row['Migration_Category'])
        
        return ' | '.join(profile_parts)
    
    enriched_df['Risk_Profile'] = enriched_df.apply(create_risk_profile, axis=1)
    
    # ========================================================================
    # INTERVENTION PRIORITY SCORING
    # ========================================================================
    print("Calculating intervention priority scores...")
    
    def calculate_priority_score(row):
        """Calculate intervention priority (0-100, higher = more urgent)"""
        score = 0
        
        # Base score from UIR (inverse - lower UIR = higher priority)
        uir = row.get('UIR', 0.2)
        if uir < 0.05:
            score += 40
        elif uir < 0.1:
            score += 35
        elif uir < 0.15:
            score += 30
        elif uir < 0.2:
            score += 25
        
        # Settlement type
        if row['Settlement_Type'] == 'Remote Rural':
            score += 20
        elif row['Settlement_Type'] == 'Rural':
            score += 10
        
        # Tribal/Forest (infrastructure challenges)
        if row['is_tribal']:
            score += 15
        elif row['is_forest_hilly']:
            score += 12
        
        # Low literacy (awareness challenge)
        if row['is_low_literacy']:
            score += 15
        
        # Migration (continuity challenge)
        if row['is_migration_source']:
            score += 10
        
        return min(score, 100)  # Cap at 100
    
    enriched_df['Intervention_Priority_Score'] = enriched_df.apply(
        calculate_priority_score, axis=1
    )
    
    def categorize_priority(score):
        """Convert score to priority category"""
        if score >= 70:
            return 'Critical Priority'
        elif score >= 50:
            return 'High Priority'
        elif score >= 30:
            return 'Medium Priority'
        else:
            return 'Low Priority'
    
    enriched_df['Priority_Level'] = enriched_df['Intervention_Priority_Score'].apply(
        categorize_priority
    )
    
    priority_dist = enriched_df['Priority_Level'].value_counts()
    print(f"Priority Level Distribution:\n{priority_dist}\n")
    
    # ========================================================================
    # DISPLAY SAMPLE RESULTS
    # ========================================================================
    print("="*80)
    print("SAMPLE ENRICHED PINCODES (First 15)")
    print("="*80)
    
    display_cols = [
        'state', 'district', 'pincode',
        'Settlement_Type', 'Tribal_Forest_Type',
        'Literacy_Category', 'Migration_Category',
        'UIR', 'Priority_Level', 'Risk_Profile'
    ]
    
    # Filter to columns that exist
    display_cols = [col for col in display_cols if col in enriched_df.columns]
    
    print(enriched_df[display_cols].head(15).to_string(index=False))
    
    # ========================================================================
    # PATTERN ANALYSIS
    # ========================================================================
    print("\n" + "="*80)
    print("PATTERN ANALYSIS")
    print("="*80)
    
    print("\nTop 10 Most Common Risk Profiles:")
    risk_counts = enriched_df['Risk_Profile'].value_counts().head(10)
    for profile, count in risk_counts.items():
        print(f"  {count:4d}  {profile}")
    
    print("\n\nCritical Priority Pincodes by State:")
    critical = enriched_df[enriched_df['Priority_Level'] == 'Critical Priority']
    if len(critical) > 0:
        critical_by_state = critical.groupby('state').size().sort_values(ascending=False).head(10)
        for state, count in critical_by_state.items():
            print(f"  {state:25s}  {count:4d} pincodes")
    else:
        print("  No critical priority pincodes found")
    
    # ========================================================================
    # SAVE ENRICHED DATA
    # ========================================================================
    if output_path is None:
        output_path = input_path.replace('.csv', '_enriched.csv')
    
    print("\n" + "="*80)
    print("SAVING ENRICHED DATA")
    print("="*80)
    
    enriched_df.to_csv(output_path, index=False)
    print(f"✓ Enriched data saved to: {output_path}")
    
    # Save critical priorities separately
    critical_path = output_path.replace('_enriched.csv', '_critical_priority.csv')
    if len(critical) > 0:
        critical.sort_values('Intervention_Priority_Score', ascending=False).to_csv(
            critical_path, index=False
        )
        print(f"✓ Critical priorities saved to: {critical_path}")
    
    # ========================================================================
    # SUMMARY STATISTICS
    # ========================================================================
    print("\n" + "="*80)
    print("ENRICHMENT SUMMARY")
    print("="*80)
    print(f"\n✓ Total pincodes enriched: {len(enriched_df)}")
    print(f"✓ States covered: {enriched_df['state'].nunique()}")
    print(f"✓ Districts covered: {enriched_df['district'].nunique()}")
    print(f"\n✓ Tribal-dominated pincodes: {enriched_df['is_tribal'].sum()}")
    print(f"✓ Forest/Hilly pincodes: {enriched_df['is_forest_hilly'].sum()}")
    print(f"✓ Low literacy pincodes: {enriched_df['is_low_literacy'].sum()}")
    print(f"✓ Migration-affected pincodes: {(enriched_df['is_migration_source'] | enriched_df['is_migration_destination']).sum()}")
    print(f"✓ Remote rural pincodes: {enriched_df['is_remote_rural'].sum()}")
    print(f"\n✓ Critical priority pincodes: {(enriched_df['Priority_Level'] == 'Critical Priority').sum()}")
    print(f"✓ High priority pincodes: {(enriched_df['Priority_Level'] == 'High Priority').sum()}")
    
    print("\n" + "="*80)
    print("ENRICHMENT COMPLETE!")
    print("="*80)
    
    return enriched_df

# ============================================================================
# RUN THE ENRICHMENT
# ============================================================================

if __name__ == "__main__":
    input_file = r'D:\Project\Hackathons\Aadhar_Hackathon\high_risk_pincodes.csv'
    
    enriched_data = enrich_high_risk_pincodes(input_file)
    
    print("\n✅ Enrichment process completed successfully!")
    print("\nYou can now use the enriched data for:")
    print("  • Targeted intervention planning")
    print("  • Resource allocation")
    print("  • Policy recommendations")
    print("  • Geographic prioritization")
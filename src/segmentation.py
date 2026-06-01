import numpy as np
import pandas as pd


MARKET_COLUMNS = [
    'income_level',
    'young_demand_level',
    'senior_demand_level',
    'competition_level',
    'growth_level',
    'cuisine_variety_level',
    'cuisine_focus_level',
    'customer_engagement_level',
    'fast_food_level',
    'healthy_level',
    'beverage_level',
    'wings_level',
    'fast_food_share',
    'healthy_share',
    'beverage_share',
    'wings_share'
]

CORE_SEGMENT_COLUMNS = [
    'customer_segment',
]

SHARE_COLUMNS = [
    'fast_food_share',
    'healthy_share',
    'beverage_share',
    'wings_share'
]

MARKET_DRIVER_COLUMNS = {
    'young_demand': 'young_demand_level',
    'senior_demand': 'senior_demand_level',
    'competition': 'competition_level',
    'growth': 'growth_level',
    'cuisine_variety': 'cuisine_variety_level',
    'customer_engagement': 'customer_engagement_level'
}

FOOD_DRIVER_COLUMNS = {
    'fast_food_market': ('fast_food_level', 'fast_food_share'),
    'healthy_market': ('healthy_level', 'healthy_share'),
    'beverage_market': ('beverage_level', 'beverage_share'),
    'wings_market': ('wings_level', 'wings_share')
}

LEVEL_LABELS = {
    'Income Level': ('income_level', 'strength'),
    'Growth Level': ('growth_level', 'growth'),
    'Competition Level': ('competition_level', 'strength'),
    'Cuisine Variety Level': ('cuisine_variety_level', 'strength'),
    'Cuisine Focus Level': ('cuisine_focus_level', 'strength'),
    'Customer Engagement Level': ('customer_engagement_level', 'strength'),
    'Fast Food Level': ('fast_food_level', 'strength'),
    'Healthy Level': ('healthy_level', 'strength'),
    'Beverage Level': ('beverage_level', 'strength'),
    'Wings Level': ('wings_level', 'strength')
}

FOOD_TYPE_PATTERNS = {
    'wings': r'[^,]*(?:wings|chicken wings|wings joint)[^,]*',
    'asian': r'[^,]*(?:asian|chinese|japanese|sushi|thai|poke)[^,]*',
    'mexican': r'[^,]*(?:mexican|taco|burrito)[^,]*',
    'seafood': r'[^,]*(?:seafood|fish|cajun)[^,]*',
    'chicken': r'[^,]*(?:chicken|fried chicken|chicken shop)[^,]*',
    'healthy': (
        r'[^,]*(?:healthy|salad|vegetarian|vegan|juice|smoothie|'
        r'mediterranean)[^,]*'
    ),
    'beverage': (
        r'[^,]*(?:coffee|juice|smoothie|bar|bars|cocktail|'
        r'cocktail bar|cocktail bars)[^,]*'
    ),
    'fast_food': r'[^,]*(?:fast food|qsr|burger|sandwich)[^,]*'
}


def split_tags(value):
    if value != value:
        return []
    return [
        tag.strip().lower()
        for tag in str(value).split(',')
        if tag.strip()
    ]


def row_tags(row):
    tags = []
    for column in ['style', 'cuisines', 'restaurant_type']:
        tags.extend(split_tags(row.get(column)))
    return tags


def tag_matches_keyword(tag, keyword):
    if keyword == 'bar':
        return tag in ['bar', 'bars', 'cocktail bar', 'cocktail bars']
    if keyword == 'cafe':
        return tag in ['cafe', 'cafes', 'coffee shop', 'coffee shops']
    if keyword == 'qsr':
        return tag == 'qsr'
    return keyword in tag


def has_any(tags, keywords):
    return any(
        tag_matches_keyword(tag, keyword)
        for tag in tags
        for keyword in keywords
    )

# Classifies restaurant into food categories.
def restaurant_food_type(row):
    tags = row_tags(row)
    food_type_keywords = {
        'wings': ['wings', 'chicken wings', 'wings joint'],
        'asian': ['asian', 'chinese', 'japanese', 'sushi', 'thai', 'poke'],
        'mexican': ['mexican', 'taco', 'burrito'],
        'seafood': ['seafood', 'fish', 'cajun'],
        'chicken': ['chicken', 'fried chicken', 'chicken shop'],
        'healthy': [
            'healthy',
            'salad',
            'vegetarian',
            'vegan',
            'juice',
            'smoothie',
            'mediterranean'
        ],
        'beverage': ['coffee', 'cafe', 'juice', 'smoothie', 'bar', 'cocktail'],
        'fast_food': ['fast food', 'qsr', 'burger', 'sandwich']
    }

    for food_type, keywords in food_type_keywords.items():
        if has_any(tags, keywords):
            return food_type

    return 'general'


FOOD_TYPE_LABELS = {
    'wings': 'Wings-focused restaurant',
    'asian': 'Asian cuisine restaurant',
    'mexican': 'Mexican cuisine restaurant',
    'seafood': 'Seafood-focused restaurant',
    'chicken': 'Chicken-focused restaurant',
    'healthy': 'Health-oriented restaurant',
    'beverage': 'Beverage-led restaurant',
    'fast_food': 'Fast-service restaurant',
    'general': 'General restaurant'
}


PRICE_CONTEXT_LABELS = {
    'premium': 'higher-income trade area',
    'balanced': 'mid-market trade area',
    'value': 'value-oriented trade area'
}


SIGNAL_LABELS = {
    'young_demand': 'younger-customer demand',
    'senior_demand': 'older-customer demand',
    'competition': 'competitive intensity',
    'growth': 'local growth',
    'cuisine_variety': 'broad local cuisine variety',
    'customer_engagement': 'customer engagement',
    'fast_food_market': 'fast-service concentration',
    'healthy_market': 'health-oriented food concentration',
    'beverage_market': 'beverage-led food concentration',
    'wings_market': 'wings and chicken-wing concentration'
}


def strength_labels(values):
    return np.select(
        [values >= 0.67, values <= 0.33],
        ['High', 'Low'],
        default='Medium'
    )


def growth_labels(values):
    return np.select(
        [values >= 0.67, values <= 0.33],
        ['Growing', 'Declining'],
        default='Stable'
    )


def age_demand_labels(young_demand, senior_demand):
    return np.select(
        [
            (young_demand >= 0.67) & (young_demand >= senior_demand),
            (senior_demand >= 0.67) & (senior_demand > young_demand),
            (young_demand <= 0.33) & (senior_demand <= 0.33)
        ],
        [
            'Young Demand High',
            'Senior Demand High',
            'Low Age Demand'
        ],
        default='Balanced Age Demand'
    )


def build_tag_text(df):
    tag_text = pd.Series('', index=df.index, dtype='object')
    for column in ['style', 'cuisines', 'restaurant_type']:
        values = df.get(column, pd.Series('', index=df.index))
        tag_text = tag_text.str.cat(
            values.fillna('').astype(str).str.lower(),
            sep=','
        )
    return tag_text


def restaurant_categories(df):
    tag_text = build_tag_text(df)
    masks = [
        tag_text.str.contains(pattern, regex=True, na=False)
        for pattern in FOOD_TYPE_PATTERNS.values()
    ]
    food_types = np.select(
        masks,
        list(FOOD_TYPE_PATTERNS.keys()),
        default='general'
    )
    return pd.Series(food_types, index=df.index).map(FOOD_TYPE_LABELS)


def describe_cluster(profile):

    drivers = {
        'Young Demand': profile['young_demand_level'],
        'Senior Demand': profile['senior_demand_level'],
        'Competition': profile['competition_level'],
        'Growth': profile['growth_level'],
        'Cuisine Variety': profile['cuisine_variety_level'],
        'Customer Engagement': profile['customer_engagement_level'],
        'Fast Food': profile['fast_food_level'],
        'Healthy Food': profile['healthy_level'],
        'Beverages': profile['beverage_level'],
        'Wings': profile['wings_level'],
        'Income': profile['income_level']
    }

    primary_driver = max(
        drivers,
        key=drivers.get
    )

    if (
        profile['senior_demand_level'] >= 0.7
        and profile['income_level'] >= 0.7
    ):
        return 'Comfort Dining Market'

    if (
        profile['young_demand_level'] >= 0.7
        and profile['fast_food_level'] >= 0.6
    ):
        return 'Youth Convenience Market'

    if (
        profile['fast_food_level'] >= 0.65
        and profile['income_level'] <= 0.4
    ):
        return 'Family Value Market'

    if profile['wings_level'] >= 0.7:
        return 'Social Dining Market'

    if (
        profile['beverage_level'] >= 0.7
        and profile['customer_engagement_level'] >= 0.6
    ):
        return 'Social Dining Market'

    if (
        profile['competition_level'] >= 0.8
        and profile['growth_level'] >= 0.65
    ):
        return 'Premium Dining Market'

    if (
        profile['income_level'] >= 0.85
        and profile['customer_engagement_level'] >= 0.65
    ):
        return 'Premium Dining Market'

    if (
        profile['healthy_level'] >= 0.75
        and profile['income_level'] >= 0.65
    ):
        return 'Healthy Lifestyle Market'

    if profile['cuisine_variety_level'] >= 0.7:
        return 'Youth Convenience Market'

    if (
        profile['senior_demand_level'] >= 0.6
        and profile['healthy_level'] <= 0.4
    ):
        return 'Comfort Dining Market'

    if (
        profile['wings_level'] >= 0.6
        and profile['beverage_level'] >= 0.6
    ):
        return 'Social Dining Market'

    if primary_driver == 'Healthy Food':
        return 'Healthy Lifestyle Market'

    if primary_driver == 'Fast Food':
        return 'Family Value Market'

    if primary_driver == 'Wings':
        return 'Social Dining Market'

    if primary_driver == 'Beverages':
        return 'Social Dining Market'

    if primary_driver == 'Cuisine Variety':
        return 'Youth Convenience Market'

    if primary_driver == 'Young Demand':
        return 'Youth Convenience Market'

    if primary_driver == 'Senior Demand':
        return 'Comfort Dining Market'

    return 'Healthy Lifestyle Market'


def assign_market_segments(df, cluster_profile):
    cluster_names = {
        cluster: describe_cluster(profile)
        for cluster, profile
        in cluster_profile.iterrows()
    }

    df = df.copy()
    df['market_segment'] = (
        df['cluster_id']
        .map(cluster_names)
    )

    return df


def _customer_segment_from_row(row):
    market_segment = str(row.get('market_segment', '')).strip()
    if market_segment:
        return market_segment

    market_segment_label = str(row.get('market_segment_label', '')).strip()
    if market_segment_label:
        return market_segment_label

    income = str(row.get('income_level', '')).lower().strip()
    age_group = str(row.get('age_group', '')).lower().strip()
    area_type = str(row.get('area_type', '')).lower().strip()

    if income in {'high', 'premium'} and age_group in {'young', 'millennial'}:
        return 'Affluent Professional'
    if age_group in {'young', 'millennial'} and area_type == 'urban':
        return 'Urban Convenience'
    if income in {'low', 'budget'} and age_group in {'young', 'millennial'}:
        return 'Budget Student'
    if 'senior' in age_group:
        return 'Senior Community'
    if area_type == 'suburban':
        return 'Suburban Family'
    return 'Balanced Regional Consumer'


def assign_core_segments(df):
    df = df.copy()
    if 'customer_segment' not in df.columns:
        df['customer_segment'] = df.apply(_customer_segment_from_row, axis=1)
    else:
        df['customer_segment'] = df['customer_segment'].fillna('').astype(str)
        missing = df['customer_segment'].str.strip() == ''
        if missing.any():
            df.loc[missing, 'customer_segment'] = df.loc[missing].apply(
                _customer_segment_from_row,
                axis=1,
            )
    return df


def assign_menu_analysis(df, market_data=None, geo_column=None):

    if geo_column is None:
        geo_column = 'ZCTA' if 'ZCTA' in df.columns else 'FIPS'

    if market_data is None:
        market_values = df.reindex(columns=MARKET_COLUMNS)
    else:
        market_values = df[[geo_column]].merge(
            market_data[[geo_column] + MARKET_COLUMNS],
            on=geo_column,
            how='left'
        )[MARKET_COLUMNS]

    market_values.index = df.index

    market_values = market_values.fillna({
        column: 0
        for column in SHARE_COLUMNS
    }).fillna(0.5)

    market_drivers = market_values[
        list(MARKET_DRIVER_COLUMNS.values())
    ].rename(columns={
        column: driver
        for driver, column in MARKET_DRIVER_COLUMNS.items()
    })
    food_drivers = market_values[
        [level for level, _ in FOOD_DRIVER_COLUMNS.values()]
    ].rename(columns={
        level: driver
        for driver, (level, _) in FOOD_DRIVER_COLUMNS.items()
    })

    for driver, (_, share_column) in FOOD_DRIVER_COLUMNS.items():
        food_drivers.loc[market_values[share_column] < 0.15, driver] = -1

    food_signals = food_drivers.idxmax(axis=1)
    food_signal_values = food_drivers.max(axis=1)
    market_signals = market_drivers.idxmax(axis=1)
    primary_signals = food_signals.where(food_signal_values >= 0, market_signals)

    signal_value_columns = {
        **MARKET_DRIVER_COLUMNS,
        **{
            driver: level
            for driver, (level, _) in FOOD_DRIVER_COLUMNS.items()
        }
    }
    signal_source_columns = primary_signals.map(signal_value_columns)
    signal_columns = list(dict.fromkeys(signal_value_columns.values()))
    signal_strength_values = np.select(
        [signal_source_columns.eq(column) for column in signal_columns],
        [market_values[column] for column in signal_columns],
        default=0.5
    )

    df['restaurant category'] = restaurant_categories(df)
    df['price positioning'] = np.select(
        [
            market_values['income_level'] >= 0.67,
            market_values['income_level'] <= 0.33
        ],
        [
            PRICE_CONTEXT_LABELS['premium'],
            PRICE_CONTEXT_LABELS['value']
        ],
        default=PRICE_CONTEXT_LABELS['balanced']
    )
    df['market trend'] = primary_signals.map(
        SIGNAL_LABELS
    ).fillna(primary_signals)
    df['menu_signal_strength'] = strength_labels(signal_strength_values)
    df['Age Demand Level'] = age_demand_labels(
        market_values['young_demand_level'],
        market_values['senior_demand_level']
    )

    for output_column, (source_column, label_type) in LEVEL_LABELS.items():
        if label_type == 'growth':
            df[output_column] = growth_labels(market_values[source_column])
        else:
            df[output_column] = strength_labels(market_values[source_column])

    return df


def assign_menu_strategy(df, market_data=None, geo_column=None):
    return assign_menu_analysis(df, market_data, geo_column)


def assign_market_intelligence(df, market_data=None, geo_column=None):
    return assign_menu_analysis(df, market_data, geo_column)

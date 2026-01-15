"""
Constants for Indian geography and mappings.
"""

# List of Indian states and union territories
INDIAN_STATES = [
    "Andhra Pradesh",
    "Arunachal Pradesh",
    "Assam",
    "Bihar",
    "Chhattisgarh",
    "Goa",
    "Gujarat",
    "Haryana",
    "Himachal Pradesh",
    "Jharkhand",
    "Karnataka",
    "Kerala",
    "Madhya Pradesh",
    "Maharashtra",
    "Manipur",
    "Meghalaya",
    "Mizoram",
    "Nagaland",
    "Odisha",
    "Punjab",
    "Rajasthan",
    "Sikkim",
    "Tamil Nadu",
    "Telangana",
    "Tripura",
    "Uttar Pradesh",
    "Uttarakhand",
    "West Bengal",
    # Union Territories
    "Andaman and Nicobar Islands",
    "Chandigarh",
    "Dadra and Nagar Haveli and Daman and Diu",
    "Delhi",
    "Jammu and Kashmir",
    "Ladakh",
    "Lakshadweep",
    "Puducherry",
]

# State centroids (approximate lat/lng)
STATE_CENTROIDS = {
    "Andhra Pradesh": {"lat": 15.9129, "lng": 79.7400},
    "Arunachal Pradesh": {"lat": 28.2180, "lng": 94.7278},
    "Assam": {"lat": 26.2006, "lng": 92.9376},
    "Bihar": {"lat": 25.0961, "lng": 85.3131},
    "Chhattisgarh": {"lat": 21.2787, "lng": 81.8661},
    "Goa": {"lat": 15.2993, "lng": 74.1240},
    "Gujarat": {"lat": 22.2587, "lng": 71.1924},
    "Haryana": {"lat": 29.0588, "lng": 76.0856},
    "Himachal Pradesh": {"lat": 31.1048, "lng": 77.1734},
    "Jharkhand": {"lat": 23.6102, "lng": 85.2799},
    "Karnataka": {"lat": 15.3173, "lng": 75.7139},
    "Kerala": {"lat": 10.8505, "lng": 76.2711},
    "Madhya Pradesh": {"lat": 22.9734, "lng": 78.6569},
    "Maharashtra": {"lat": 19.7515, "lng": 75.7139},
    "Manipur": {"lat": 24.6637, "lng": 93.9063},
    "Meghalaya": {"lat": 25.4670, "lng": 91.3662},
    "Mizoram": {"lat": 23.1645, "lng": 92.9376},
    "Nagaland": {"lat": 26.1584, "lng": 94.5624},
    "Odisha": {"lat": 20.9517, "lng": 85.0985},
    "Punjab": {"lat": 31.1471, "lng": 75.3412},
    "Rajasthan": {"lat": 27.0238, "lng": 74.2179},
    "Sikkim": {"lat": 27.5330, "lng": 88.5122},
    "Tamil Nadu": {"lat": 11.1271, "lng": 78.6569},
    "Telangana": {"lat": 18.1124, "lng": 79.0193},
    "Tripura": {"lat": 23.9408, "lng": 91.9882},
    "Uttar Pradesh": {"lat": 26.8467, "lng": 80.9462},
    "Uttarakhand": {"lat": 30.0668, "lng": 79.0193},
    "West Bengal": {"lat": 22.9868, "lng": 87.8550},
    "Andaman and Nicobar Islands": {"lat": 11.7401, "lng": 92.6586},
    "Chandigarh": {"lat": 30.7333, "lng": 76.7794},
    "Dadra and Nagar Haveli and Daman and Diu": {"lat": 20.4283, "lng": 72.8397},
    "Delhi": {"lat": 28.7041, "lng": 77.1025},
    "Jammu and Kashmir": {"lat": 33.7782, "lng": 76.5762},
    "Ladakh": {"lat": 34.1526, "lng": 77.5771},
    "Lakshadweep": {"lat": 10.5667, "lng": 72.6417},
    "Puducherry": {"lat": 11.9416, "lng": 79.8083},
}

# Major district centroids (sample - can be expanded)
DISTRICT_CENTROIDS = {
    # Rajasthan
    "Jaipur": {"lat": 26.9124, "lng": 75.7873, "state": "Rajasthan"},
    "Jodhpur": {"lat": 26.2389, "lng": 73.0243, "state": "Rajasthan"},
    "Udaipur": {"lat": 24.5854, "lng": 73.7125, "state": "Rajasthan"},
    "Ajmer": {"lat": 26.4499, "lng": 74.6399, "state": "Rajasthan"},
    "Kota": {"lat": 25.2138, "lng": 75.8648, "state": "Rajasthan"},
    "Bikaner": {"lat": 28.0229, "lng": 73.3119, "state": "Rajasthan"},
    "Alwar": {"lat": 27.5530, "lng": 76.6346, "state": "Rajasthan"},
    "Sikar": {"lat": 27.6094, "lng": 75.1398, "state": "Rajasthan"},
    "Sawai Madhopur": {"lat": 26.0173, "lng": 76.3466, "state": "Rajasthan"},
    "Ganganagar": {"lat": 29.9038, "lng": 73.8772, "state": "Rajasthan"},
    
    # Maharashtra
    "Mumbai": {"lat": 19.0760, "lng": 72.8777, "state": "Maharashtra"},
    "Pune": {"lat": 18.5204, "lng": 73.8567, "state": "Maharashtra"},
    "Nagpur": {"lat": 21.1458, "lng": 79.0882, "state": "Maharashtra"},
    "Nashik": {"lat": 19.9975, "lng": 73.7898, "state": "Maharashtra"},
    "Aurangabad": {"lat": 19.8762, "lng": 75.3433, "state": "Maharashtra"},
    "Thane": {"lat": 19.2183, "lng": 72.9781, "state": "Maharashtra"},
    "Ratnagiri": {"lat": 16.9902, "lng": 73.3120, "state": "Maharashtra"},
    
    # Karnataka
    "Bengaluru Urban": {"lat": 12.9716, "lng": 77.5946, "state": "Karnataka"},
    "Bengaluru Rural": {"lat": 13.1300, "lng": 77.5700, "state": "Karnataka"},
    "Mysuru": {"lat": 12.2958, "lng": 76.6394, "state": "Karnataka"},
    "Mangaluru": {"lat": 12.9141, "lng": 74.8560, "state": "Karnataka"},
    "Hubli-Dharwad": {"lat": 15.3647, "lng": 75.1240, "state": "Karnataka"},
    "Tumakuru": {"lat": 13.3379, "lng": 77.1173, "state": "Karnataka"},
    "Hassan": {"lat": 13.0068, "lng": 76.1004, "state": "Karnataka"},
    "Shivamogga": {"lat": 13.9299, "lng": 75.5681, "state": "Karnataka"},
    
    # Tamil Nadu
    "Chennai": {"lat": 13.0827, "lng": 80.2707, "state": "Tamil Nadu"},
    "Coimbatore": {"lat": 11.0168, "lng": 76.9558, "state": "Tamil Nadu"},
    "Madurai": {"lat": 9.9252, "lng": 78.1198, "state": "Tamil Nadu"},
    "Tiruchirappalli": {"lat": 10.7905, "lng": 78.7047, "state": "Tamil Nadu"},
    "Salem": {"lat": 11.6643, "lng": 78.1460, "state": "Tamil Nadu"},
    "Karur": {"lat": 10.9571, "lng": 78.0766, "state": "Tamil Nadu"},
    
    # Uttar Pradesh
    "Lucknow": {"lat": 26.8467, "lng": 80.9462, "state": "Uttar Pradesh"},
    "Kanpur Nagar": {"lat": 26.4499, "lng": 80.3319, "state": "Uttar Pradesh"},
    "Ghaziabad": {"lat": 28.6692, "lng": 77.4538, "state": "Uttar Pradesh"},
    "Agra": {"lat": 27.1767, "lng": 78.0081, "state": "Uttar Pradesh"},
    "Varanasi": {"lat": 25.3176, "lng": 82.9739, "state": "Uttar Pradesh"},
    "Aligarh": {"lat": 27.8974, "lng": 78.0880, "state": "Uttar Pradesh"},
    "Gorakhpur": {"lat": 26.7606, "lng": 83.3732, "state": "Uttar Pradesh"},
    "Bahraich": {"lat": 27.5747, "lng": 81.5965, "state": "Uttar Pradesh"},
    "Firozabad": {"lat": 27.1591, "lng": 78.3957, "state": "Uttar Pradesh"},
    "Maharajganj": {"lat": 27.1310, "lng": 83.5627, "state": "Uttar Pradesh"},
    "Ghazipur": {"lat": 25.5878, "lng": 83.5742, "state": "Uttar Pradesh"},
    
    # Bihar
    "Patna": {"lat": 25.5941, "lng": 85.1376, "state": "Bihar"},
    "Gaya": {"lat": 24.7914, "lng": 85.0002, "state": "Bihar"},
    "Sitamarhi": {"lat": 26.5903, "lng": 85.4912, "state": "Bihar"},
    "Madhubani": {"lat": 26.3620, "lng": 86.0800, "state": "Bihar"},
    "Purbi Champaran": {"lat": 26.6474, "lng": 84.8697, "state": "Bihar"},
    "Madhepura": {"lat": 25.9241, "lng": 86.7916, "state": "Bihar"},
    "Bhojpur": {"lat": 25.4670, "lng": 84.4451, "state": "Bihar"},
    "Vaishali": {"lat": 25.6867, "lng": 85.2167, "state": "Bihar"},
    
    # Gujarat
    "Ahmedabad": {"lat": 23.0225, "lng": 72.5714, "state": "Gujarat"},
    "Surat": {"lat": 21.1702, "lng": 72.8311, "state": "Gujarat"},
    "Vadodara": {"lat": 22.3072, "lng": 73.1812, "state": "Gujarat"},
    "Rajkot": {"lat": 22.3039, "lng": 70.8022, "state": "Gujarat"},
    "Gandhinagar": {"lat": 23.2156, "lng": 72.6369, "state": "Gujarat"},
    "Anand": {"lat": 22.5645, "lng": 72.9289, "state": "Gujarat"},
    "Patan": {"lat": 23.8500, "lng": 72.1167, "state": "Gujarat"},
    "Sabarkantha": {"lat": 23.6013, "lng": 73.0514, "state": "Gujarat"},
    "Valsad": {"lat": 20.5992, "lng": 72.9342, "state": "Gujarat"},
    
    # West Bengal
    "Kolkata": {"lat": 22.5726, "lng": 88.3639, "state": "West Bengal"},
    "Howrah": {"lat": 22.5958, "lng": 88.2636, "state": "West Bengal"},
    "Hooghly": {"lat": 22.9000, "lng": 88.3833, "state": "West Bengal"},
    "Paschim Medinipur": {"lat": 22.4200, "lng": 87.3200, "state": "West Bengal"},
    
    # Andhra Pradesh
    "Visakhapatnam": {"lat": 17.6868, "lng": 83.2185, "state": "Andhra Pradesh"},
    "Vijayawada": {"lat": 16.5062, "lng": 80.6480, "state": "Andhra Pradesh"},
    "Chittoor": {"lat": 13.2172, "lng": 79.1003, "state": "Andhra Pradesh"},
    "Srikakulam": {"lat": 18.2949, "lng": 83.8935, "state": "Andhra Pradesh"},
    "Kurnool": {"lat": 15.8281, "lng": 78.0373, "state": "Andhra Pradesh"},
    
    # Telangana
    "Hyderabad": {"lat": 17.3850, "lng": 78.4867, "state": "Telangana"},
    "Warangal": {"lat": 17.9784, "lng": 79.5941, "state": "Telangana"},
    "Mulugu": {"lat": 18.1900, "lng": 79.9400, "state": "Telangana"},
    
    # Kerala
    "Thiruvananthapuram": {"lat": 8.5241, "lng": 76.9366, "state": "Kerala"},
    "Kochi": {"lat": 9.9312, "lng": 76.2673, "state": "Kerala"},
    "Thrissur": {"lat": 10.5276, "lng": 76.2144, "state": "Kerala"},
    "Wayanad": {"lat": 11.6854, "lng": 76.1320, "state": "Kerala"},
    
    # Odisha
    "Bhubaneswar": {"lat": 20.2961, "lng": 85.8245, "state": "Odisha"},
    "Cuttack": {"lat": 20.4625, "lng": 85.8830, "state": "Odisha"},
    "Nayagarh": {"lat": 20.1286, "lng": 85.0962, "state": "Odisha"},
    "Dhenkanal": {"lat": 20.6700, "lng": 85.6000, "state": "Odisha"},
    
    # Punjab
    "Ludhiana": {"lat": 30.9010, "lng": 75.8573, "state": "Punjab"},
    "Amritsar": {"lat": 31.6340, "lng": 74.8723, "state": "Punjab"},
    "Rupnagar": {"lat": 30.9660, "lng": 76.5330, "state": "Punjab"},
    
    # Haryana
    "Faridabad": {"lat": 28.4089, "lng": 77.3178, "state": "Haryana"},
    "Gurugram": {"lat": 28.4595, "lng": 77.0266, "state": "Haryana"},
    "Mahendragarh": {"lat": 28.2833, "lng": 76.1500, "state": "Haryana"},
    
    # Jammu and Kashmir
    "Srinagar": {"lat": 34.0837, "lng": 74.7973, "state": "Jammu and Kashmir"},
    "Jammu": {"lat": 32.7266, "lng": 74.8570, "state": "Jammu and Kashmir"},
    "Punch": {"lat": 33.7667, "lng": 74.0833, "state": "Jammu and Kashmir"},
    
    # Meghalaya
    "East Khasi Hills": {"lat": 25.4670, "lng": 91.3662, "state": "Meghalaya"},
    "Shillong": {"lat": 25.5788, "lng": 91.8933, "state": "Meghalaya"},
}

# Age group labels
AGE_GROUPS = {
    "enrollment": ["0-5", "5-17", "18+"],
    "demographic": ["5-17", "17+"],
    "biometric": ["5-17", "17+"],
}

# View mode options
VIEW_MODES = ["daily", "monthly", "quarterly"]

# Severity levels for anomalies
SEVERITY_LEVELS = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]

# Risk levels
RISK_LEVELS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

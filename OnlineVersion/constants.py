"""
Constants related to Claim Scoring
"""

"""
These represent a mapping of generic data to individual tag data to individual coaters.
"""

#define the list of input tags
INPUT_TAGS = ['numClaims',
              'numSales',
              'dollarPerClaim',
              'dollarPerSale',
              'claimRatio',
              'INVENTORY_STYLE_CD',
              'INVENTORY_SIZE_CD',
              'INVENTORY_BACKING_CD',
              'INVENTORY_COLOR_CD'
              ]

OUTPUT_TAGS = ['CLAIM_NUM',
               'INVENTORY_STYLE_CD',
               'INVENTORY_SIZE_CD',
               'INVENTORY_BACKING_CD',
               'INVENTORY_COLOR_CD',
               'DIM_DIVISION_CUSTOMER_GK',
               'CLAIM',
               'dollarClaims',
               'numClaims',
               'numSales',
               'dollarSales',
               'dollarPerClaim',
               'dollarPerSale',
               'claimRatio',
               'Pred',
               'Score']
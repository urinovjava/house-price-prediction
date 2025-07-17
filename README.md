# house-price-prediction
### **A machine learning regression project to predict house sale prices based on features like area, number of bedrooms and bathrooms, floor count, build year, location type, and other physical or categorical characteristics.**

At the beginning of the analysis, I reviewed all available columns in the dataset, which included: `date`, `price`, `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, `condition`, `sqft_above`, `sqft_basement`, `yr_built`, `yr_renovated`, `street`, `city`, `statezip`, and `country`. I decided to drop the `street` and `statezip` columns early on, since they were not important for the modetl. I also dropped the `country` column after discovering that all rows indicated the same value (“USA”), making it irrelevant for prediction.

Next, I processed the `date` column. Instead of using the full timestamp, I extracted two new features: the month of sale (`date_month`) and the day of the week (`date_weekday`). This allowed me to capture potential seasonal or behavioral effects on housing prices. Then, `date` column was dropped.

For the `city` column I used target encoding, a technique that replaces each city name with the average house price in that city. This created a new numeric feature named `city_encoded`, which allowed the model to use location information.

During exploratory data analysis, I used correlation matrices to examine relationships between variables. It became apparent that some features were highly correlated. For instance, `sqft_living` had a very high correlation with both `sqft_above` and `bathrooms`. To reduce multicollinearity and simplify the model, I chose to drop `sqft_above` and `bedrooms`.

I performed a correlation analysis between all numerical features and the target variable price. Based on this analysis, I selected only the features that demonstrated relatively strong correlations with the target. Specifically, I retained the following columns: sqft_living, sqft_above, bathrooms, view, sqft_basement, bedrooms, waterfront, floors, and the encoded version of city. Features with low correlation coefficients such as sqft_lot, condition, date_month, date_weekday, yr_built, and yr_renovated were dropped from the dataset as they were most likely unsignificant.

I then split the dataset into training and testing sets and applied standardization to the numerical features to ensure that the scale of the variables did not bias the model. The first model I trained was a simple **Linear Regression**. This model achieved a Mean Absolute Error (MAE) of approximately **139,828**, and a Root Mean Squared Error (RMSE) of about **252,765**. I also evaluated the model using 10-fold cross-validation, where the average RMSE was roughly **222,000**. 

Following this, I trained a **Random Forest Regressor** model in the hope of capturing non-linear relationships. However, while the MAE slightly improved to **137,865**, the RMSE increased to **268,682**, and the cross-validation results showed greater variance and generally worse performance. This suggests that the Random Forest model was potentially overfitting or not generalizing well on this dataset.

To further test the importance of the `city_encoded` feature, I dropped it from the dataset and retrained a new Linear Regression model. As expected, model performance deteriorated: the MAE rose to **174,890**, and the RMSE increased to **278,578**. Additionally, the cross-validation scores became unstable, with several extremely negative values — an indicator of unreliable predictions. This experiment confirmed that the `city_encoded` variable played a significant role in improving model accuracy.

In conclusion, the best-performing model in this project was the Linear Regression model that included  `city_encoded` column along with selected numeric predictors. Although I experimented with more complex algorithms like Random Forest and additional feature removal, they did not yield better results. Therefore, I chose to finalize the project with the Linear Regression model due to its consistent and interpretable performance.


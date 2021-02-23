import itertools
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import shapiro, mannwhitneyu
from statsmodels.stats.api import DescrStatsW


pd.set_option('display.float_format', lambda x: '%.5f' % x)
pd.set_option('display.max_columns', None)

df_ = pd.read_csv("datasets/pricing.csv", sep=";")

df = df_.copy()
df.head()
###############################
# DATA PREPROCESSING
###############################
df.info()

df["price"].describe([0.01, 0.25, 0.5, 0.75, 0.99]).T

# count     3448.00000
# mean      3254.47577
# std      25235.79901
# min         10.00000
# 1%          30.00000
# 25%         31.89044
# 50%         34.79854
# 75%         41.53621
# 99%     201436.46420
# max     201436.99126
# Name: price, dtype: float64

df.isnull().sum()
# There are no NA values


def calculate_data_will_be_affected(dataframe, column,
                                    low_quantile=0.0, up_quantile=1.0):
    """
    This function stands for preliminary information about calculation what
    percentage of data is affected by manipulation that we will make.

    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
        DataFrame that we want to analyse for affection according quantiles that
        we specified.
    column: str
        Column name that we want to see affection according to quantiles that we
        specified.
    low_quantile: float
        The number to be assigned as the low threshold value of the relevant
        column.
    up_quantile: float
        The number to be assigned as the up threshold value of the relevant
        column.

    Returns
    -------

    """
    ratio_for_up_limit = (len(dataframe[dataframe[column] > dataframe[column].
                              quantile(up_quantile)]) /
                          dataframe[column].shape[0]) * 100
    ratio_for_low_limit = (len(dataframe[dataframe[column] < dataframe[column].
                           quantile(low_quantile)]) /
                           dataframe[column].shape[0]) * 100
    ratio_of_affected_data = round(ratio_for_up_limit + ratio_for_low_limit, 2)
    print("When we do reassign operation in the ({} - {}) range, %{} of data "
          "will be affected in this operation.".format(low_quantile,
                                                       up_quantile,
                                                       ratio_of_affected_data))


calculate_data_will_be_affected(df, 'price', 0.05, 0.95)
# When we do reassign operation in the (0.05 - 0.95) range, %5.34 of data will
# be affected in this operation.


def outlier_thresholds(dataframe, col_name, low_quantile, up_quantile):
    """
    This function stands for determine the threshold values of given dataframe
    according to given column name, lower quantile value and upper quantile
    value.

    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
        DataFrame that we want to determine thresholds for given column name.
    col_name: str
        Column name to determine outlier.
    low_quantile: float
        The number to be assigned as the low threshold value of the relevant
        column.
    up_quantile: float
        The number to be assigned as the up threshold value of the relevant
        column.

    Returns: tuple
        Determined threshold values.
    -------

    """
    quartile1 = dataframe[col_name].quantile(low_quantile)
    quartile3 = dataframe[col_name].quantile(up_quantile)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    """
    This function stands for checking if a column of a dataframe have outlier or
    not according to results of outlier_thresholds function.

    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
        DataFrame that we want to check for outlier values.
    col_name: str
        Column name to check for outlier values.

    Returns: bool
        True, False
    -------

    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, 0.05, 0.95)
    if dataframe[(dataframe[col_name] > up_limit) |
                 (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


check_outlier(df, "price")


def replace_with_thresholds(dataframe, col_name, low_threshold, up_threshold):
    """
    This function stands for replacing given columns of dataframe according to
    given lower threshold and upper threshold values via using outlier_threshold
    function.

    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
        DataFrame that we want to manipulate.
    col_name: str
        Column name which low and high levels will be arranged according
        to given thresholds.
    low_threshold: float
        The number to be assigned as the low threshold value of the relevant
        column.
    up_threshold: float
        The number to be assigned as the up threshold value of the relevant
        column.

    Returns
    -------

    """
    low_limit, up_limit = outlier_thresholds(dataframe, col_name,
                                             low_threshold, up_threshold)
    if low_limit > 0:
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit
    else:
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit


replace_with_thresholds(df, 'price', 0.05, 0.95)

df["price"].describe([0.01, 0.25, 0.5, 0.75, 0.99]).T

# count   3448.00000
# mean      43.68247
# std       28.21820
# min       10.00000
# 1%        30.00000
# 25%       31.89044
# 50%       34.79854
# 75%       41.53621
# 99%      187.44554
# max      187.44554
# Name: price, dtype: float64


def plot_columns(dataframe, title):
    """
    This function stands for plotting graphs of 'category_id' and 'price'
    columns for visual examination.

    Parameters
    ----------
    dataframe: DataFrame that we want to plot its graph.
    title: Title for plotted graph.

    Returns
    -------

    """
    sns.boxplot(x=dataframe['category_id'], y=dataframe['price'])
    plt.title(title)
    plt.xlabel('Category ID')
    plt.ylabel('Price')
    plt.show()


plot_columns(df_, 'Price Distribution Before Reassign With Thresholds')
plot_columns(df, 'Price Distribution After Reassign With Thresholds')


df_.groupby('category_id').agg({'price': ['count', 'mean', 'median', 'std',
                                          'min', 'max']})
df.groupby('category_id').agg({'price': ['count', 'mean', 'median', 'std',
                                         'min', 'max']})

#############################
# A/B Testing
#############################


def test_normality(dataframe, iteration_column, target_column):
    """
    This function stands for testing the target column of given dataframe
    according to normal distribution is valid on it or not via using shapiro
    test.

    H0: Normal distribution
    H1: Not normal distribution
    If p-value > 0.05 we can not reject H0 value for normality test.

    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
        DataFrame that we want to test its target column.
    iteration_column: str
        Column name that we will iterate over to test for normality according to
        target column.
    target_column: str
        Column name that we will calculate normality over its values.

    Returns: tuple
        Normal distributed ids and not normal distributed ids according to
        shapiro test.
    -------

    """
    normals = []
    not_normals = []
    category_ids = [cat_id for cat_id in dataframe[iteration_column].unique()]
    for id_ in category_ids:
        ttest, p_value = shapiro(dataframe.
                                 loc[dataframe[iteration_column] == id_,
                                     target_column])
        if p_value >= 0.05:
            normals.append(id_)
        else:
            not_normals.append(id_)
    return normals, not_normals


normals_list, not_normals_list = test_normality(df, 'category_id', 'price')
# According to shapiro test H0 is rejected. Therefore we could leap over
# variance homogeneity test.

# Since price values of all category_id's not normally distributed we will make
# nonparametric A/B Test with mannwhitneyu.


def make_nonparametric_ab_test(dataframe, iteration_column,
                               target_column, not_normal_ids_list):
    """
    This function stands for testing means of two group statistically
    significant or not.
    H0: There is no statistically significant difference between means of given
    two groups.
    H1: There is statistically significant difference between means of given two
    groups.

    If p-value > 0.05, then we can not reject H0 hypothesis.

    Makes calculations over these hypothesis and assumption.

    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
        Dataframe that we will make nonparametric AB Test over it.
    iteration_column: str
        Column name that we will iterate over to test for nonparametric AB Test
        according to target column.
    target_column: str
        Column name that will be test over it.
    not_normal_ids_list: list
        A list that contains id's that can not pass normality test.
    Returns: tuple
        Returns a tuple that contains lists of rejected and not rejected pairs
        according to nonparametric test.
    -------

    """
    rejected_pairs = []
    not_rejected_pairs = []
    category_list = list(itertools.combinations(not_normal_ids_list, 2))
    for i in category_list:
        ttest, p_value = mannwhitneyu(dataframe.
                                      loc[dataframe[iteration_column] == i[0],
                                          target_column],
                                      dataframe.
                                      loc[dataframe[iteration_column] == i[1],
                                          target_column])
        if p_value >= 0.05:
            not_rejected_pairs.append(i)
        else:
            rejected_pairs.append(i)
    return rejected_pairs, not_rejected_pairs


rejecteds, not_rejecteds = make_nonparametric_ab_test(df, 'category_id',
                                                      'price', not_normals_list)
similar_price_categories = set()

for pair in not_rejecteds:
    for price in pair:
        similar_price_categories.add(price)

different_price_categories = set(df['category_id'].unique()) -\
                             similar_price_categories
# 1- Item'in fiyatı kategorilere göre farklılık göstermekte midir? İstatistiki
# olarak ifade ediniz.

# Kategorilere göre fiyat farklılığı da fiyat benzerliği de gözlenebiliyor.
# Bunun için nonparametric AB Test sonucundan çıkan benzer ve farklı kategoriler
# ayrımına bakabiliriz.

# similar_price_categories: {201436, 361254, 675201, 874521}
# different_price_categories: {326584, 489756}


def compare_results(dataframe, iteration_column,
                    target_column, list_to_compare):
    """
    This function stands for giving a table to visualize the conclusion.
    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
        Dataframe that we made test over it.
    iteration_column: str
        Column name that we iterated over in the test.
    target_column:
        Column name that we test over it.
    list_to_compare: list
        A list which contains results that we want to compare and visualize it.
    Returns:
    -------

    """
    for i in list_to_compare:
        compare_df = pd.concat([dataframe[lambda x: x[iteration_column] ==
                                          i[0]],
                                dataframe[lambda x: x[iteration_column] ==
                                          i[1]]],
                               axis=0)
        print(compare_df.groupby(iteration_column).
              agg({target_column: ['mean', 'median', 'std']}))


compare_results(df, 'category_id', 'price', rejecteds)
compare_results(df, 'category_id', 'price', not_rejecteds)


# Since the data is not normal distributed calculations will be over median
# value, lower bound value and upper bound value
def calculate_incomes(dataframe, target_column, lower_price, upper_price,
                      median_price, calculation_type):
    """

    Parameters
    ----------
    dataframe: pandas.core.frame.DataFrame
        DataFrame that we want to calculate income values over lower, upper and
        median prices.
    target_column: str
        Column name that we want to calculate income over it.
    lower_price: numpy.float64
        Lower price bound according to confidence interval for specific group or
        all data from dataframe.
    upper_price: numpy.float64
        Upper price bound according to confidence interval for specific group or
        all data from dataframe.
    median_price: float
        Median price according to specific group or all data from dataframe.
    calculation_type: str
        The name of method that we want to give for calculations.

    Returns: pandas.core.frame.DataFrame
        Gives a DataFrame that includes information about income and price
        values.
    -------

    """
    median_price_income = dataframe.loc[dataframe[target_column] >=
                                        median_price].shape[0] * median_price
    lower_price_income = dataframe.\
        loc[dataframe[target_column] >= lower_price].\
        shape[0] * lower_price
    upper_price_income = dataframe.\
        loc[dataframe[target_column] >= upper_price].\
        shape[0] * upper_price
    df_last = pd.DataFrame({'median_price_income': median_price_income,
                            'median_price': median_price,
                            'median_count': dataframe.
                           loc[dataframe['price'] >= median_price].shape[0],
                            'lower_price_income': lower_price_income,
                            'lower_price': lower_price,
                            'lower_count': dataframe.loc[dataframe['price'] >=
                                                         lower_price].shape[0],
                            'upper_price_income': upper_price_income,
                            'upper_price': upper_price,
                            'upper_count': dataframe.loc[dataframe['price'] >=
                                                         upper_price].shape[0]
                            }, index=[0])
    df_last.insert(0, 'calculation_type', value=[calculation_type])

    return df_last


median_price_all = df['price'].median()
lower_all_data, upper_all_data = DescrStatsW(df['price']).tconfint_mean()
all_data_income_df = calculate_incomes(df, 'price', lower_all_data,
                                       upper_all_data, median_price_all,
                                       'median_all_data')

#   calculation_type  median_price_income  median_price  median_count  \
# 0  median_all_data          59992.69045      34.79854          1724
#    lower_price_income  lower_price  lower_count  upper_price_income  \
# 0         34192.21159     42.74026          800         31951.26973
#    upper_price  upper_count
# 0     44.62468          716


# 2- İlk soruya bağlı olarak item'ın fiyatı ne olmalıdır? Nedenini açıklayınız?
# Hem yapılan gelir hesabı hem de verinin normal dağılmadığı düşünüldüğünde
# ürün fiyatı için alınacak kararın ortalama yerine medyan üzerinden
# belirlenmesinin daha sağlıklı sonuç vereceği görülüyor. Gelir hesabı tüm
# kategorileri düşünerek yapıldığında en fazla gelirin medyan değeri üzerinden
# alındığı görünüyor.Ancak yapılan nonparametric teste göre kategorilere göre
# fiyat farklılığı olduğu da anlaşılıyor. Bu durumda benzer fiyat kategorileri
# için benzer fiyat kategorilerini bir grup, farklı fiyat kategorilerinin her
# birini birer grup olarak düşünüp gruplara median üzerinden fiyat belirleme
# stratejisi ile devam edebilriz.

# similar_price_categories: {201436, 361254, 675201, 874521}
# different_price_categories: {326584, 489756}

median_similar_prices = df.loc[(df['category_id'] != 489756) &
                               (df['category_id'] != 326584), 'price'].median()
lower_without_489756_326584, upper_without_489756_326584 = \
    DescrStatsW(df.loc[(df['category_id'] != 489756) &
                       (df['category_id'] != 326584), 'price']).tconfint_mean()
similar_prices_df = calculate_incomes(df, 'price', lower_without_489756_326584,
                                      upper_without_489756_326584,
                                      median_similar_prices,
                                      'median_similar_prices')

#         calculation_type  median_price_income  median_price  median_count  \
# 0  median_similar_prices          66286.96422      34.39905          1927
#    lower_price_income  lower_price  lower_count  upper_price_income  \
# 0         36883.55459     38.94779          947         35962.10514
#    upper_price  upper_count
# 0     41.38332          869

median_for_489756 = df.loc[df['category_id'] == 489756, 'price'].median()
lower_for_489756, upper_for_489756 = \
    DescrStatsW(df.loc[df['category_id'] == 489756, 'price']).tconfint_mean()
df_for_489756 = calculate_incomes(df, 'price', lower_for_489756,
                                  upper_for_489756, median_for_489756,
                                  'median_for_489756')

#     calculation_type  median_price_income  median_price  median_count  \
# 0  median_for_489756          45970.16166      35.63578          1290
#    lower_price_income  lower_price  lower_count  upper_price_income  \
# 0         30738.25976     46.08435          667         28451.25429
#    upper_price  upper_count
# 0     49.05389          580

median_for_326584 = df.loc[df['category_id'] == 326584, 'price'].median()
lower_for_326584, upper_for_326584 = \
    DescrStatsW(df.loc[df['category_id'] == 326584, 'price']).tconfint_mean()
df_for_326584 = calculate_incomes(df, 'price', lower_for_326584,
                                  upper_for_326584, median_for_326584,
                                  'median_for_326584')

#     calculation_type  median_price_income  median_price  median_count  \
# 0  median_for_326584          82958.15612      31.74824          2613
#    lower_price_income  lower_price  lower_count  upper_price_income  \
# 0         70240.63684     33.88357         2073         37418.13385
#    upper_price  upper_count
# 0     39.59591          945

median_without_489756 = df.loc[df['category_id'] != 489756, 'price'].median()
lower_without_489756, upper_without_489756 = \
    DescrStatsW(df.loc[df['category_id'] != 489756, 'price']).tconfint_mean()
df_without_489756 = calculate_incomes(df, 'price', lower_without_489756,
                                      upper_without_489756,
                                      median_without_489756,
                                      'median_without_489756')

#         calculation_type  median_price_income  median_price  median_count  \
# 0  median_without_489756          67754.14973      34.21927          1980
#    lower_price_income  lower_price  lower_count  upper_price_income  \
# 0         37537.95745     38.73886          969         36304.69899
#    upper_price  upper_count
# 0     41.02226          885

median_without_326584 = df.loc[df['category_id'] != 326584, 'price'].median()
lower_without_326584, upper_without_326584 = \
    DescrStatsW(df.loc[df['category_id'] != 326584, 'price']).tconfint_mean()
df_without_326584 = calculate_incomes(df, 'price', lower_without_326584,
                                      upper_without_326584,
                                      median_without_326584,
                                      'median_without_326584')

#         calculation_type  median_price_income  median_price  median_count  \
# 0  median_without_326584          59192.81842      34.86032          1698
#    lower_price_income  lower_price  lower_count  upper_price_income  \
# 0         33679.04191     43.01282          783         31518.13962
#    upper_price  upper_count
# 0     44.96168          701

frames = [all_data_income_df, similar_prices_df, df_for_489756, df_for_326584,
          df_without_489756, df_without_326584]

result_df = pd.concat(frames, ignore_index=True)

result_df.sort_values('median_price_income', ascending=False).head(1)

#     calculation_type  median_price_income  median_price  median_count  \
# 3  median_for_326584          82958.15612      31.74824          2613
#    lower_price_income  lower_price  lower_count  upper_price_income  \
# 3         70240.63684     33.88357         2073         37418.13385
#    upper_price  upper_count
# 3     39.59591          945

result_df.sort_values('median_count', ascending=False).head(1)

#     calculation_type  median_price_income  median_price  median_count  \
# 3  median_for_326584          82958.15612      31.74824          2613
#    lower_price_income  lower_price  lower_count  upper_price_income  \
# 3         70240.63684     33.88357         2073         37418.13385
#    upper_price  upper_count
# 3     39.59591          945

# Soru 2'nin cevabının devamı;
# Cevabın başında belirtilen "benzer fiyat kategorileri için benzer fiyat
# kategorilerini bir grup, farklı fiyat kategorilerinin her birinin birer grup
# olarak düşünerek median üzerinden fiyat belirlemek" işini yaptıktan sonra
# gruplar gelirlerine göre sıralandığında category_id: 326584 için 31.74824 ürün
# fiyatı ile gelirin maksimum olduğu (82958.15612) görülmektedir. Dolayısıyla
# başlangıçtaki kullanıcı sayısından büyük bir kayıp yaşamadan, maksimum
# kullanıcı sayısı (2613) ile maksimum geliri sağlayan fiyat olan 31.74824 ürün
# fiyatı olarak belirlenebilir.

# 3- Fiyat konusunda "hareket edebilir olmak" istenmektedir. Fiyat stratejisi
# için karar destek sistemi oluşturunuz.

sns.scatterplot(x='median_price', y='median_price_income',
                hue='calculation_type', data=result_df)
plt.xlabel('Item Price')
plt.ylabel('Income')
plt.title('Price-Income and Calculation Method Relationship', color='black')
plt.show()

# Grafikten de görülebileceği üzere ürün fiyatı arttıkça gelirimiz düşüyor.
# Medyan değerlerine göre fiyat aralığımız ise 31.74824 - 35.63578 değerleri
# arasında değişmektedir. 31.74824 değeri için maksimum gelir, 35.6378 değeri
# için medyana göre minimum gelir elde ettiğimiz düşünülürse yöneticilerin kabul
# edilebilir gördükleri getiriye göre bu fiyatlar arasında hareket etme
# konusunda bir esnekliğe sahip olmuş olur.

# 4- Olası fiyat değişiklikleri için item satın almalarını ve gelirlerini simüle
# ediniz.
# Fiyat değişikliklerine göre oluşabilecek senaryolar tablolarda ve grafiklerde
# mevcut.

import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

cv_len = 160000

def get_cv_test(train):
    test = train.filter(pl.col("user_id") > cv_len)  # 本地验证集
    test_label = (test.sort(["user_id", "click_timestamp"], descending=[True, True]).group_by("user_id").agg(pl.all().first()))
    test_label = test_label["user_id", "click_article_id"]
    test_label = test_label.rename({"click_article_id": "label"})
    test = (test.sort(["user_id", "click_timestamp"], descending=[False, True]).with_columns([pl.col("user_id").cum_count(reverse=False).over("user_id").alias("rank")]))
    test = test.filter(pl.col("rank") > 1)
    test = test["user_id", "click_article_id","click_timestamp"]
    return test, test_label

def get_testa():
    test = pl.read_csv("../../dataset/testA_click_log.csv")
    test = test.sort(["user_id", "click_timestamp"], descending=[False, True])
    test = test["user_id", "click_article_id", "click_timestamp"]
    return test

def itemcf(submit=0):
    art_info = pl.read_csv("../../dataset/articles.csv")
    art_info = art_info.with_columns(pl.col("article_id").alias("click_article_id")).drop("article_id")
    train = pl.read_csv("../../dataset/train_click_log.csv")
    train = train.join(art_info, on="click_article_id", how="left")
    if submit == 0:
        test, test_label = get_cv_test(train)
        train = train.filter(pl.col("user_id") <= cv_len)
    else:
        test = get_testa()

    train = train.with_columns(pl.count("click_article_id").over("user_id").alias("user_click_count"))                    # 统计每个用户的点击数
    train = train.with_columns((1.0 / 2**((pl.col("click_timestamp") - pl.col("created_at_ts") + 1)/(3600*1000*24*5)).abs()).alias("timedis"))
    # train = train.with_columns(((-(pl.col("click_timestamp") - pl.col("created_at_ts"))/(3600*1000*24*4)).exp()).alias("timedis"))

    train_tmp = train.unique(subset=["user_id", "click_article_id"])
    article_counts = train_tmp["click_article_id"].value_counts()                                                         # 每个文章的总惦记数目，反映热门程度
    joined = train.join(train, on="user_id")                                                                              # 自己合并自己，形成商品-商品对
    result = joined.filter((pl.col("click_timestamp") < pl.col("click_timestamp_right")) & (pl.col("click_article_id") != pl.col("click_article_id_right")))                                                                                                                     # 去除重复的商品对，去除未来的时间
    result = result.drop("user_click_count_right").drop("user_id")                                                        # 删去没用的信息
    result = result.join(article_counts, on="click_article_id", how="left")                                               # 给合并后的表格的左物品，添加物品数目
    result = result.join(article_counts, left_on="click_article_id_right", right_on="click_article_id", how="left")       # 给合并后的表格的右物品，添加物品数目
    result = result.with_columns(((pl.col("count") * pl.col("count_right") + 2).sqrt()).alias("coef")).drop("count").drop("count_right")   # 各自次数乘积，开根号
    result = result.with_columns((1.0 / 2**((pl.col("click_timestamp") - pl.col("click_timestamp_right") + 1)/(3600*1000)).abs()).alias("trend")) # 两篇文章的相隔时间，以小时数的指数递减
    result = result.with_columns(((pl.col("category_id") == pl.col("category_id_right")).cast(float)*0.5+1).alias("type_weight"))
    result = result.with_columns((1.0 / 2**((pl.col("words_count") - pl.col("words_count_right") + 1)/(500)).abs()).alias("words"))

    result = result.with_columns((1.0 / (pl.col("user_click_count") + 1).log()).alias("iif"))                             # 用户点击数目对数倒数，一个人假如点击次数太多，那权重应该相应降低
    result = result.with_columns((1.0 / (pl.col("user_click_count") + 1)**0.3).alias("iif2"))                             # 更合理的方式
    result = result.with_columns((1.0 / pl.col("coef") * pl.col("iif2") / pl.col("type_weight") * pl.col("timedis") * pl.col("timedis_right") * pl.col("words")).alias("weight"))                                  # 合并权重  * pl.col("trend")
    result = result.drop("iif").drop("trend").drop("coef")                                                                # 删去不用的信息
    result = result.group_by(["click_article_id", "click_article_id_right"]).agg(pl.sum("weight"))                        # 按照“商品-商品对”把权重叠加作为相似度
    result = result.sort(["click_article_id", "weight"], descending=True)                                                 # 按照相似度降序排列
    result = result.group_by("click_article_id").head(200)

    if submit == 0 :
        cv_test(result, test, test_label)
    else:
        cv_test(result, test)



def cv_test(result, test_sel, test_label = None):
    test = test_sel
    test_sel= test_sel.join(result[["click_article_id", "click_article_id_right", "weight"]], on="click_article_id", how="left")
    test_sel = test_sel.fill_null(0)
    test_sel = test_sel.sort(["user_id","weight"], descending=True)
    itemcf = test_sel.group_by(["user_id", "click_article_id_right"]).agg(pl.sum("weight"))
    itemcf_out = itemcf.sort(["user_id","weight"], descending=True)

    test = test.rename({"click_article_id": "click_article_id_right"})
    last_time = test.group_by("user_id").agg(pl.col("click_timestamp").first())

    art_info = pl.read_csv("../../dataset/articles.csv")
    art_info = art_info.with_columns(pl.col("article_id").alias("click_article_id_right")).drop("article_id")
    itemcf_out = itemcf_out.join(art_info, on="click_article_id_right", how="left")

    itemcf_out = itemcf_out.join(last_time, on="user_id", how="left")
    itemcf_out = itemcf_out.filter(pl.col("created_at_ts") < pl.col("click_timestamp"))

    tmp = itemcf_out.group_by('user_id').agg(pl.mean("words_count"))
    tmp = tmp.rename({"words_count": "user_mean_words"})
    itemcf_out = itemcf_out.join(tmp, on = 'user_id', how = "left")

    itemcf_out = itemcf_out.with_columns(((pl.col("words_count") - pl.col("user_mean_words")).abs()).alias("dwords")).drop('words_count').drop('user_mean_words')
    itemcf_out = itemcf_out.with_columns((1.0 / 2 ** (pl.col('dwords')/1000)).alias("dwords_weight")).drop('dwords')

    itemcf_out = itemcf_out.with_columns(((pl.col("click_timestamp") - pl.col("created_at_ts"))/(1000*60*60)).alias("dt")).drop('created_at_ts').drop('click_timestamp').drop('category_id')
    itemcf_out = itemcf_out.with_columns((1.0 / 2 ** (pl.col('dt')/40)).alias("dt_weight")).drop('dt')
    itemcf_out = itemcf_out.with_columns((pl.col("weight") * pl.col("dt_weight")).alias("weight_new")).drop('weight').drop('dt_weight').drop("dwords_weight")
    itemcf_out = itemcf_out.sort(["user_id","weight_new"], descending=True)
    itemcf_out = itemcf_out.group_by("user_id").head(5)

    missing_users = test.select("user_id").filter(~pl.col("user_id").is_in(itemcf_out["user_id"]))
    empty_rows = missing_users.with_columns([
        pl.lit(None).cast(itemcf_out.schema[col]).alias(col)
        for col in itemcf_out.columns if col != "user_id"
    ])
    itemcf_out = pl.concat([itemcf_out, empty_rows])
    print(itemcf_out)

    if isinstance(test_label, pl.DataFrame):
        # art_info = art_info.with_columns(pl.col("click_article_id_right").alias("label")).drop("click_article_id_right")
        # tmp = test_label.join(art_info, on="label", how="left")
        # tmp = tmp.join(last_time, on="user_id", how="left")
        # tmp = tmp.with_columns(((pl.col("click_timestamp") - pl.col("created_at_ts")) / (1000 * 60 * 60)).alias("dt"))
        # print(tmp)
        df = test_label.join(itemcf_out, on="user_id")
        df = df.with_columns((pl.col("label") == pl.col("click_article_id_right")).alias("is_correct"))                       # 判断预测的和真实的是否一致
        df = df.group_by("user_id").agg(pl.any("is_correct").alias("true"))                                                   # 按用户分组，只有有一个匹配上就算预测成功
        predict = df.filter(pl.col("true")).height                                                                            # 统计成功预测的用户数
        acc = predict / (test_label.height)                                                                                   # 计算准确率
        print(acc, test_label.height)
    else:
        df_grouped = (
            itemcf_out.group_by("user_id")
            .agg(pl.col("click_article_id_right").alias("articles"))
            .with_columns([
                pl.col("articles").list.slice(0, 5).list.to_struct()
            ])
            .unnest("articles")
        )
        df_final = df_grouped.rename({f"field_{i}": f"article_{i + 1}" for i in range(5)})
        df_final = df_final.sort(["user_id"], descending=False)
        df_final.write_csv("recommendations.csv", include_header=True)

        print(df_final)

itemcf(submit=1)


library(stringr)
library(data.table)
library(dummies)
library(ggplot2)

reddit_comments_path = "/home/cbrom/workspace/datasets/big_data/reddit_comments.csv"

# reddit_comments <- read.csv(reddit_comments_path, nrows=1000)
# length is 1236782
reddit_comments <- read.csv(reddit_comments_path, nrows=1000000)

nrow(reddit_comments)

head(reddit_comments)

# drop when title _cosine is nan
df <- reddit_comments[complete.cases(reddit_comments[, 'title_cosine']), ]

getmode <- function(values) {
    uniqvalues <- unique(values)
    uniqvalues[which.max(tabulate(match(values, uniqvalues)))]
}

# impute with mode of parent column score
parent_score_impute <- getmode(subset(df, (!is.na(df$parent_score)))$parent_score)
comment_tree_root_score_impute <- getmode(subset(df, (!is.na(df$comment_tree_root_score)))$comment_tree_root_score)
time_since_comment_tree_root_impute <- getmode(subset(df, (!is.na(df$time_since_comment_tree_root)))$time_since_comment_tree_root)
parent_cosine_impute <- 0
parent_euc_impute <- 0

# replace impute values
# parent_score
# comment_tree_root_score
# time_since_comment_tree_root
# parent_cosine
# parent_euc
df[is.na(df$parent_score)==TRUE, ]$parent_score <- parent_score_impute
df[is.na(df$comment_tree_root_score) == TRUE, ]$comment_tree_root_score <- comment_tree_root_score <- comment_tree_root_score_impute
df[is.na(df$time_since_comment_tree_root)==TRUE, ]$parent_score <- time_since_comment_tree_root_impute
df[is.na(df$parent_cosine)==TRUE, ]$parent_cosine <- parent_cosine_impute
df[is.na(df$parent_euc)==TRUE, ]$parent_euc <- parent_euc_impute


bool_cols <- c('over_18', 'is_edited', 'is_quoted', 'is_selftext')

cat_cols <- c('subreddit', 'distinguished','hour_of_comment', 'weekday')

numeric_cols <- c('gilded', 'controversiality', 'upvote_ratio','time_since_link',
                'depth', 'no_of_linked_sr', 'no_of_linked_urls', 'parent_score',
                'comment_tree_root_score', 'time_since_comment_tree_root',
                'subjectivity', 'senti_neg', 'senti_pos', 'senti_neu',
                'senti_comp', 'no_quoted', 'time_since_parent', 'word_counts',
                'no_of_past_comments', 'parent_cosine','parent_euc',
                'title_cosine', 'title_euc','link_score')

df[,numeric_cols] = apply(df[,numeric_cols], 2, function(x) as.double(as.character(x)));

features <- do.call(c, list(bool_cols, cat_cols, numeric_cols))

new_df <- df[, features]

head(new_df)

## change boolean columns
new_df$over_18 <- as.integer(as.logical(new_df$over_18))
new_df$is_edited <- as.integer(as.logical(new_df$is_edited))
new_df$is_quoted <- as.integer(as.logical(new_df$is_quoted))
new_df$is_selftext <- as.integer(as.logical(new_df$is_selftext))
# new_df$is_flair <- as.character(new_df$is_flair)
# new_df$is_flair_css <- as.character(new_df$is_flair_css)

dummy_df <- dummy.data.frame(new_df, names=cat_cols)
# dummy_df$score <- df$score

tbplot <- new_df
tbplot$score <- df$score

ggplot(data = tbplot, mapping = aes(x = title_cosine, y = score)) +
geom_point()

ggplot(data = tbplot, mapping = aes(x = gilded, y = score)) +
geom_point()

ggplot(data = tbplot, mapping = aes(x = controversiality, y = score)) +
geom_point()

ggplot(data = tbplot, mapping = aes(x = upvote_ratio, y = score)) +
geom_point()

ggplot(data = tbplot, mapping = aes(x = time_since_link, y = score)) +
geom_point()

ggplot(data = tbplot, mapping = aes(x = depth, y = score)) +
geom_point()

ggplot(data = tbplot, mapping = aes(x = no_of_linked_sr, y = score)) +
geom_point()

ggplot(data = tbplot, mapping = aes(x = no_of_linked_urls, y = score)) +
geom_point()

ggplot(data = tbplot, mapping = aes(x = parent_score, y = score)) +
geom_point()

ggplot(data = tbplot, mapping = aes(x = comment_tree_root_score, y = score)) +
geom_point()

ggplot(data = tbplot, mapping = aes(x = time_since_comment_tree_root, y = score)) +
geom_point()

ggplot(data = tbplot, mapping = aes(x = subjectivity, y = score)) +
geom_point()

ggplot(data = tbplot, mapping = aes(x = senti_neg, y = score)) +
geom_point()

ggplot(data = tbplot, mapping = aes(x = senti_pos, y = score)) +
geom_point()

ggplot(data = tbplot, mapping = aes(x = senti_neu, y = score)) +
geom_point()

ggplot(data = tbplot, mapping = aes(x = no_of_past_comments, y = score)) +
geom_point()

ggplot(data = tbplot, mapping = aes(x = parent_cosine, y = score)) +
geom_point()

ggplot(data = tbplot, mapping = aes(x = parent_euc, y = score)) +
geom_point()

ggplot(data = tbplot, mapping = aes(x = title_cosine, y = score)) +
geom_point()

ggplot(data = tbplot, mapping = aes(x = title_euc, y = score)) +
geom_point()

ggplot(data = tbplot, mapping = aes(x = link_score, y = score)) +
geom_point()

target <- 'score'

x <- dummy_df
y <- df[, target]

head(x)

train_len <- floor(0.75 * nrow(x))
set.seed(0)
train_index <- sample(seq_len(nrow(x)), size=train_len)

train_x <- x[train_index, ]
test_x <- x[-train_index, ]
train_y <- y[train_index]
test_y <- y[-train_index]

head(train_x)

linearMod <- lm(train_y ~ .,train_x)

summary(linearMod)

newLinearMod <- lm( train_y ~ over_18 + is_edited + is_quoted + subredditfood + subredditgaming + subredditmovies + subredditscience + distinguishedmoderator + hour_of_comment5 + hour_of_comment6 + hour_of_comment12 + hour_of_comment13 + hour_of_comment14 + hour_of_comment15 + weekday2 + gilded + controversiality + upvote_ratio + time_since_link + depth + no_of_linked_sr + no_of_linked_urls + parent_score + comment_tree_root_score + time_since_comment_tree_root  + subjectivity + time_since_parent + word_counts + no_of_past_comments + parent_cosine + parent_euc + title_euc +link_score, train_x)

summary(newLinearMod)

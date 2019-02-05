
library(DBI)
library(stringr)
library(wordVectors)

sql_con <- dbConnect(RSQLite::SQLite(), "/home/cbrom/workspace/2015-02.db")

get_time <- function(){
    return(as.numeric(Sys.time()))
}

start <- get_time()
sql_data <- dbGetQuery(sql_con, "SELECT parent, comment from parent_reply")
paste('Total time to read:', get_time() - start, sep=" ")

paste('Total rows:', length(sql_data$parent), sep=" ")

head(sql_data)

caps = "([A-Z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)\\."
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
starters = "(Mr|Mrs|Ms|Dr|He\\s|She\\s|It\\s|They\\s|Their\\s|Our\\s|We\\s|But\\s|However\\s|That\\s|This\\s|Wherever)"
websites = "\\.(com|edu|gov|io|me|net|org)"
digits = "([0-9])"

split_into_sentences <- function(text){
  text = gsub("\n|\r\n"," ", text)
  text = gsub(prefixes, "\\1<prd>", text)
  text = gsub(websites, "<prd>\\1", text)
  text = gsub('www\\.', "www<prd>", text)
  text = gsub("Ph.D.","Ph<prd>D<prd>", text)
  text = gsub(paste0("\\s", caps, "\\. "), " \\1<prd> ", text)
  text = gsub(paste0(acronyms, " ", starters), "\\1<stop> \\2", text)
  text = gsub(paste0(caps, "\\.", caps, "\\.", caps, "\\."), "\\1<prd>\\2<prd>\\3<prd>", text)
  text = gsub(paste0(caps, "\\.", caps, "\\."), "\\1<prd>\\2<prd>", text)
  text = gsub(paste0(" ", suffixes, "\\. ", starters), " \\1<stop> \\2", text)
  text = gsub(paste0(" ", suffixes, "\\."), " \\1<prd>", text)
  text = gsub(paste0(" ", caps, "\\."), " \\1<prd>",text)
  text = gsub(paste0(digits, "\\.", digits), "\\1<prd>\\2", text)
  text = gsub("...", "<prd><prd><prd>", text, fixed = TRUE)
  text = gsub('\\.”', '”.', text)
  text = gsub('\\."', '\".', text)
  text = gsub('\\!"', '"!', text)
  text = gsub('\\?"', '"?', text)
  text = gsub('\\.', '.<stop>', text)
  text = gsub('\\?', '?<stop>', text)
  text = gsub('\\!', '!<stop>', text)
  text = gsub('<prd>', '.', text)
  sentence = strsplit(text, "<stop>\\s*")
  return(sentence)
}

clean_text <- function(pairs, out_name){
    total_rows <- nrow(pairs)
    start <- get_time()
    out_file <- file(out_name, 'a')
    for (row in 1: 50) {
#         for (row in 1: nrow(pairs)){
        parent <- pairs$parent[row]
        comment <- pairs$comment[row]

        # remove newLines
        par_no_tabs <- gsub("\\t", " ", parent)
        com_no_tabs <- gsub("\\t", " ", comment)

        # normalize to alpha and dot(.)
        par_alpha <- gsub("[^a-zA-Z\\.]", " ", par_no_tabs)
        com_alpha <- gsub("[^a-zA-Z\\.]", " ", com_no_tabs)

        # strip
        par_strip <- gsub("^\\s+|\\s+$", "", par_alpha)
        com_strip <- gsub("^\\s+|\\s+$", "", com_alpha)
        
        # change multi space to 1
        par_multi_space <- gsub(" +", " ", par_strip)
        com_multi_space <- gsub(" +", " ", com_strip)

        # lowercase
        par_clean <- tolower(par_multi_space)
        com_clean <- tolower(com_multi_space)

        # tokenize
        par_sents <- split_into_sentences(par_clean)
        for (i in {x <- 1:length(par_sents[[1]])}){
            par_sents[[1]][i] <- sub("[\\.]", "", par_sents[[1]][i])
        }

        com_sents <- split_into_sentences(com_clean)
        for (i in {x <- 1:length(com_sents[[1]])}){
            com_sents[[1]][i] <- sub("[\\.]", "", com_sents[[1]][i])
        }
        
        print(par_sents)

        if (length(par_clean) > 0 & str_count(par_clean, " ")> 0){
            for (idx in 1:length(par_sents[[1]])) {
                writeLines(par_sents[[1]][idx], out_file)
            }
        }
        if (length(com_clean) > 0 & str_count(com_clean, " ")> 0){
            for (idx in 1:length(com_clean[[1]])) {
                
                writeLines(com_sents[[1]][idx], out_file)
            }
        }

        if (row %% 500 == 0){
            total_time = get_time() - start
            paste('Completed ', round(row/total_rows), '% - ', row,
                 'rows in time', round(total_time / 60), 'min %',
                 round(total_time%%60), 'secs\r')
        }

        flush(out_file)
    }
    close(out_file)
    
}

start <- get_time()
clean_text(sql_data,  "/home/cbrom/workspace/test.txt")
paste('Total time to read:', get_time() - start, sep=" ")

start <- get_time()
num_features <- 100
min_word_count <- 40
context <- 5
downsampling <- 1e-3
print("Training model ...")
model <- # train model here
model_name <- "model_full_r"
#save moodel here
# model.init_sims if possible 

if (file.exists('r_model.bin')){
    model = read.vectors('r_model.bin')
} else {
    model <- train_word2vec(train_file = "/home/cbrom/workspace/full.txt",output_file = "./r_model.bin", vectors = num_features, window = context, threads = 4, min_count = min_word_count, negative_samples = 0)
}

model %>% closest_to("feminist")

model %>% closest_to(model[[c("fish","salmon","trout","shad","flounder","carp","roe","eels")]],50)

set.seed(10)
centers = 150
clustering = kmeans(model,centers=centers,iter.max = 40)

sapply(sample(1:centers,10),function(n) {
  names(clustering$cluster[clustering$cluster==n][1:10])
})

ingredients = c("madeira","beef","saucepan","carrots")
term_set = lapply(ingredients, 
       function(ingredient) {
          nearest_words = model %>% closest_to(model[[ingredient]],20)
          nearest_words$word
        }) %>% unlist
subset = model[[term_set,average=F]]
subset %>%
  cosineDist(subset) %>% 
  as.dist %>%
  hclust %>%
  plot

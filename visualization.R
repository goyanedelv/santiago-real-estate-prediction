
library("leaflet")
library("rgdal")
library("openxlsx")
library("dplyr")
library("htmlwidgets")

# prices data
data <- read.xlsx("data/predicted_and_actuals_full_df.xlsx")
manzanas <- as.factor(data$manzana)
prices <- data$price
prices_per_block <- data.frame(MANZENT_I = manzanas, price = prices)

# geography data
path <- "data/R13_MANZANA_IND_C17.shp.geojson"
res <- readOGR(dsn = path, layer = "MANZANA_IND_C17", encoding = "UTF-8")
res_stgo <- subset(res, MANZENT_I %in% manzanas)

# Merging geographic data with prices
res_stgo_with_prices <- merge(res_stgo, prices_per_block, duplicateGeoms = T)

pal <- colorNumeric("viridis", NULL)

map <- leaflet(res_stgo_with_prices) %>%
    addTiles() %>%
    setView(lng = -70.63, lat = -33.47, zoom = 12) %>%
    addPolygons(data = res_stgo_with_prices,
        stroke = FALSE,
        smoothFactor = 0.3,
        fillOpacity = 1,
        color = ~pal(price),
        label = ~paste0(MANZENT_I, ": ", formatC(price, big.mark = ","))) %>%
    addLegend("bottomright", pal = pal, values = ~price,
        title = "Price per squared meter",
        labFormat = labelFormat(prefix = "UF"),
        opacity = 1
    )

saveWidget(map, "data/santiago.html", selfcontained = FALSE,
    title = "Prices in Santiago")

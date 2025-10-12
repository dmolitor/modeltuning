library(hexSticker)
library(magick)

img <- image_read(here::here("man", "figures", "wizard2.jpg"))
img <- img |>
  image_convert("png") |>
  image_fill(color = "none") |>
  image_annotate(
    text = "modeltuning",
    font = "Brush Script MT",
    style = "normal",
    weight = 1000,
    size = 100,
    location = "+210+370",
    color = "gray80"
)

s <- sticker(
  filename = here::here("man", "figures", "logo.png"),
  white_around_sticker = TRUE,
  img,
  package = "",
  s_x = 1,
  s_y = 1,
  s_width = 2,
  s_height = 14,
  h_size = 3,
  h_fill = "white",
  h_color = "#A9A9A9"
) + ggplot2::theme(plot.margin = ggplot2::margin(b=0, l=0, unit="lines"))

ggplot2::ggsave(
  plot = s,
  filename = here::here("man", "figures", "logo.png"),
  width = 43.9,
  height = 50.8,
  units = "mm"
)

# Remove the background at remove.bg
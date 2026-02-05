# Spheroid Segmentation & Volume Estimation

library(shiny)
library(shinycssloaders)
library(DT)
library(imager)
library(magick)
library(abind)
library(base64enc)
library(tensorflow)
library(reticulate)
library(keras)
#install_tensorflow(envname = "r-tensorflow", method = "auto")
#reticulate::install_miniconda()
#options(reticulate.conda_environment = "r-reticulate")
# reticulate::virtualenv_create("cnn_env", python = "python3.10")
# reticulate::use_virtualenv("cnn_env", required = TRUE)
# reticulate::py_install(c("tensorflow", "keras", "tflite-support", "numpy"))


envname <- "cnn_env" 

if (Sys.info()[["sysname"]] == "Linux") {
  message("Running on Linux: using tflite-runtime")
  if (!virtualenv_exists(envname)) {
    virtualenv_create(envname = envname, python = "/usr/bin/python3") 
  }
  
  use_virtualenv(envname, required = TRUE) 
  
  reticulate::py_install(
    packages = c("numpy==1.26.4", "tflite-runtime"), 
    envname = envname, 
    pip = TRUE, 
    ignore_installed = FALSE
  )
  
  tflite_runtime <- import("tflite_runtime.interpreter", delay_load = TRUE)
  Interpreter <- tflite_runtime$Interpreter
} else {
  message("Running on Windows: using TensorFlow")
  if (!virtualenv_exists(envname))
    virtualenv_create(envname = envname)
  use_virtualenv(envname, required = TRUE)
  tf <- import("tensorflow", delay_load = TRUE)
  Interpreter <- tf$lite$Interpreter
}

model_path <- "unet_model.tflite"


sanitize_filename <- function(x) gsub("[^A-Za-z0-9_\\-\\.]", "_", x)

load_uploaded_images <- function(files, target_size = c(256, 256)) {
  if (is.null(files)) stop("File not Found")
  if (!is.data.frame(files)) {
    files <- data.frame(
      name = basename(files),
      datapath = files,
      stringsAsFactors = FALSE
    )
  }
  
  imgs <- list()
  valid_names <- character(0)
  
  for (i in seq_len(nrow(files))) {
    path <- files$datapath[i]
    fname <- files$name[i]
    im <- NULL
    try({ im <- load.image(path) }, silent = TRUE)
    if (is.null(im)) {
      try({
        im_magick <- image_read(path)
        im <- magick2cimg(im_magick)
      }, silent = TRUE)
    }
    if (!is.null(im)) {
      im <- grayscale(im)
      im <- resize(im, target_size[1], target_size[2])
      arr <- array(as.numeric(im), dim = c(1, target_size[2], target_size[1], 1))
      imgs[[length(imgs) + 1]] <- arr
      valid_names <- c(valid_names, fname)
    }
  }
  if (length(imgs) == 0) stop("No Valid File.")
  list(images = do.call(abind, c(imgs, list(along = 1))), filenames = valid_names)
}

calculate_feret <- function(mask_cimg, scale_factor = 1000 / 88.5002) {
  mask_mat <- as.matrix(mask_cimg[,,1,1] > 0.5)
  labeled <- imager::label(as.cimg(mask_mat))
  labeled_mat <- as.matrix(labeled)
  regs <- setdiff(unique(labeled_mat), 0)
  if (length(regs) == 0)
    return(list(Dmin = NA, Dmax = NA, Volume = NA, p1 = NULL, p2 = NULL))
  
  region_sizes <- table(labeled_mat[labeled_mat > 0])
  biggest <- as.numeric(names(which.max(region_sizes)))
  coords <- which(labeled_mat == biggest, arr.ind = TRUE)
  if (nrow(coords) < 2)
    return(list(Dmin = NA, Dmax = NA, Volume = NA, p1 = NULL, p2 = NULL))
  max_points <- 300
  if (nrow(coords) > max_points) {
    idx_sample <- seq(1, nrow(coords), length.out = max_points)
    coords <- coords[round(idx_sample), ]
  }
  
  dist_matrix <- as.matrix(dist(coords))
  dmax_val <- max(dist_matrix, na.rm = TRUE)
  idx <- which(dist_matrix == dmax_val, arr.ind = TRUE)[1,]
  p1 <- coords[idx[1],]; p2 <- coords[idx[2],]
  vec <- p2 - p1
  vec_norm <- vec / sqrt(sum(vec^2))
  proj <- abs((coords[,1] - p1[1]) * vec_norm[2] - (coords[,2] - p1[2]) * vec_norm[1])
  Dmax <- dmax_val * scale_factor
  Dmin <- median(proj, na.rm = TRUE) * scale_factor
  if (is.na(Dmin) || Dmin < Dmax/3) Dmin <- Dmax/3
  Volume <- (Dmin^2) * Dmax * 0.5
  list(Dmin = Dmin, Dmax = Dmax, Volume = Volume, p1 = p1, p2 = p2)
}

make_overlay <- function(orig_img, mask_cimg, p1, p2) {
  orig <- grayscale(orig_img)
  orig <- resize(orig, 256, 256)
  grad_x <- imgradient(mask_cimg, "x")
  grad_y <- imgradient(mask_cimg, "y")
  edges <- sqrt(grad_x^2 + grad_y^2) > 0.05
  edges <- resize(edges, dim(orig)[1], dim(orig)[2])
  orig_rgb <- imappend(list(orig, orig, orig), "c")
  edges_rgb <- imappend(list(edges, edges, edges), "c")
  overlay <- orig_rgb
  overlay[,,1] <- pmax(orig_rgb[,,1], edges_rgb[,,1])
  overlay[,,2] <- orig_rgb[,,2] * (1 - edges_rgb[,,2])
  overlay[,,3] <- orig_rgb[,,3] * (1 - edges_rgb[,,3])
  im_m <- image_read(as.raster(overlay))
  img_draw <- image_draw(im_m)
  if (!is.null(p1) && !is.null(p2)) {
    segments(x0 = p1[1], y0 = p1[2],
             x1 = p2[1], y1 = p2[2],
             col = "blue", lwd = 2)
  }
  
  dev.off()
  img_draw
}


ui <- shiny::fluidPage(
  shiny::titlePanel("Spheroid Segmentation & Volume Estimation"),
  shiny::sidebarLayout(
    shiny::sidebarPanel(
      shiny::fileInput("imgs", "Upload Microscopy Images", multiple = TRUE,
                       accept = c(".tif", ".tiff", ".png", ".jpg", ".jpeg")),
      shiny::numericInput("px_per_1000um", "Pixels per 1000 µm", value = 88.5002, step = 0.0001),
      shiny::actionButton("run", "Run Segmentation", class = "btn-primary"),
      shiny::hr(),
      shiny::downloadButton("dl_zip", "Download Results (ZIP)"),
      shiny::downloadButton("dl_csv", "Download Table (CSV)")
    ),
    shiny::mainPanel(
      shiny::h4("Predicted Overlays"),
      shiny::uiOutput("overlay_preview") %>% shinycssloaders::withSpinner(),
      shiny::hr(),
      DT::DTOutput("table") %>% shinycssloaders::withSpinner()
    )
  )
)


server <- function(input, output, session) {
  model_r <- shiny::reactiveVal(NULL)
  results <- shiny::reactiveVal(NULL)
  workdir <- shiny::reactiveVal(NULL)
  
  
  observe({
    if (is.null(model_r())) {
      if (Sys.info()[["sysname"]] == "Windows") {
        showNotification("Loading Keras model (.h5)...", type = "message", duration = 5)
        tryCatch({
          model <- keras::load_model_hdf5("unet_11march.h5", compile = FALSE)
          model_r(model)
          showNotification("Keras model loaded.",type= "message", duration = 5)
        }, error = function(e) {
          showNotification(paste("Error: Loading model .h5:", e$message), type = "error")
        })
      } else {
        showNotification("Loading TFlite model (.tflite)...", type = "message", duration = 5)
        tryCatch({
          tflite_runtime <- import("tflite_runtime.interpreter", delay_load = TRUE)
          interpreter <- tflite_runtime$Interpreter(model_path = "unet_model.tflite")
          interpreter$allocate_tensors()
          model_r(interpreter)
          showNotification("TFlite model loaded.", type = "message", duration = 5)
        }, error = function(e) {
          showNotification(paste("Error: Loading model .tflite:", e$message), type = "error")
        })
      }
    }
  })
  
  
  shiny::observeEvent(input$run, {
    shiny::validate(shiny::need(input$imgs, "Upload Image."))
    shiny::showNotification("Loading...")
    
    unet_model <- model_r()
    shiny::showNotification("Verifying CNN model")
    
    shiny::validate(shiny::need(!is.null(unet_model), 
                                "Error: CNN model cannot be loaded."))
    
    shiny::showNotification("Processing your image(s)", duration = 5)
    
    scale_factor <- 1000 / input$px_per_1000um
    wdir <- tempfile("run_"); dir.create(wdir)
    mask_dir <- file.path(wdir, "Predicted_masks"); dir.create(mask_dir)
    overlay_dir <- file.path(wdir, "Predicted_overlays"); dir.create(overlay_dir)
    workdir(wdir)
    
    li <- tryCatch(
      load_uploaded_images(input$imgs),
      error = function(e) {
        shiny::showNotification(paste("Error loading images:", e$message), type = "error")
        return(NULL)
      })
    if (is.null(li)) return(invisible(NULL))
    
    X <- li$images; fnames <- li$filenames
    
    Xnp <- X 
    
    preds_list <- list()
    
    if (Sys.info()[['sysname']] == "Windows") {
      
      preds_list[[1]] <- unet_model$predict(Xnp)
      
    } else {
      
      input_details <- unet_model$get_input_details()
      output_details <- unet_model$get_output_details()
      np <- import("numpy", convert = FALSE)
      
      for (i in 1:dim(Xnp)[1]) {
        
        single_image_batch <- Xnp[i, , , , drop = FALSE] 
        input_data <- np$array(single_image_batch, dtype = "float32")
        
        unet_model$set_tensor(input_details[[1]]$index, input_data)
        unet_model$invoke()
        
        preds_list[[i]] <- unet_model$get_tensor(output_details[[1]]$index)
      }
    }    
    
    
    preds <- do.call(abind::abind, c(preds_list, list(along = 1)))
    
    df <- data.frame(File = fnames, D_min_um = NA_real_,
                     D_max_um = NA_real_, Volume_um3 = NA_real_)
    for (i in seq_along(fnames)) {
      shiny::showNotification(
        paste0("Processing image ", i, "/", length(fnames)),
        type = "message"
      )
      
      base <- sanitize_filename(tools::file_path_sans_ext(basename(fnames[i])))
      
      file_path <- if (!is.null(input$imgs$datapath)) {
        input$imgs$datapath[i]
      } else if (file.exists(fnames[i])) {
        fnames[i]
      } else {
        file.path(getwd(), fnames[i])
      }
      
      if (!file.exists(file_path)) {
        stop("Image file not found: ", file_path)
      }
      
      nd <- length(dim(preds))
      mask_array <- switch(
        as.character(nd),
        "4" = preds[i, , , 1],
        "3" = preds[, , 1],
        "2" = preds,
        stop(paste("Unexpected preds dimensions:", paste(dim(preds), collapse = "x")))
      )
      mask_array <- as.array(mask_array)
      
      mask_img <- as.cimg(mask_array)
      save.image(mask_img, file.path(mask_dir, paste0(base, "_mask.png")))
      
      orig <- load.image(file_path)
      orig <- grayscale(orig)
      orig <- resize(orig, 256, 256)
      
      fer <- calculate_feret(mask_img, scale_factor)
      overlay_img <- make_overlay(orig, mask_img, fer$p1, fer$p2)
      
      image_write(
        overlay_img,
        file.path(overlay_dir, paste0(base, "_overlay.png")),
        format = "png"
      )
      
      df[i, 2:4] <- c(fer$Dmin, fer$Dmax, fer$Volume)
    }
    results(df)
  })
  
  output$table <- DT::renderDT({
    req(results())
    DT::datatable(results(), options = list(pageLength = 10, scrollX = TRUE))
  })
  
  output$overlay_preview <- shiny::renderUI({
    req(workdir())
    overlay_dir <- file.path(workdir(), "Predicted_overlays")
    imgs <- list.files(overlay_dir, full.names = TRUE, pattern = "\\.png$")
    if (length(imgs) == 0) return(NULL)
    shiny::tagList(lapply(imgs, function(p)
      shiny::tags$img(src = base64enc::dataURI(file = p, mime = "image/png"),
                      style = "max-width: 100%; margin-bottom: 12px; border: 1px solid #ccc;")))
  })
  
  output$dl_csv <- shiny::downloadHandler(
    filename = function() sprintf("Predicted_volumes_%s.csv", format(Sys.time(), "%Y%m%d_%H%M%S")),
    content = function(file) {
      req(results()); write.csv(results(), file, row.names = FALSE)
    })
  
  output$dl_zip <- shiny::downloadHandler(
    filename = function() sprintf("Masks_and_Overlays_%s.zip", format(Sys.time(), "%Y%m%d_%H%M%S")),
    content = function(file) {
      req(workdir())
      owd <- setwd(workdir()); on.exit(setwd(owd), add = TRUE)
      if (.Platform$OS.type == "windows") {
        ps <- paste0(
          "Compress-Archive -Force -Path ",
          shQuote("Predicted_masks"), ",", shQuote("Predicted_overlays"),
          " -DestinationPath ", shQuote(normalizePath(file, winslash = "\\\\", mustWork = FALSE))
        )
        system2("powershell", args = c("-NoProfile", "-Command", ps))
      } else {
        system2("zip", args = c("-r", shQuote(file),
                                shQuote("Predicted_masks"), shQuote("Predicted_overlays")))
      }
    })
}

shiny::shinyApp(ui = ui, server = server)


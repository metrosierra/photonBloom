# photonBloom

Swabian Instruments Network Time Tagger interfacing ecosystem.

- The subroutine folder contains scripts which provide abstracted (very general) functions such as cross correlation and fast histogram binning

- big_tagger, client_object, and server_object are part of the direct hardware interfacing, where big_tagger also contains sub-classed methods that use the time tagger's hardware for data processing on the fly

- Now we have a stable live plotting version (mark 2) and even a GUI coming up!!!!!

- The pipeline scripts are experimental/exploratory, where their development would lead to more established subroutine scripts being created along the way


- Dev Note: Please do NOT push raw data files onto the repository unless it is a static reference document (EG configuration JSON which should NOT be deleted). Please create a folder called output in your local clone and output datafiles there. Folders called data and output are ignored (see .gitignore)

- If data has to be shared, use onedrive, it's great!

- Also, please PULL before PUSHING as usual :D

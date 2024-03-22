# Death-and-Taxes
Data Driven Supernova Taxonomy Based on the Zwicky Transient Facility Bright Transient Survey

---

# Meeting Notes

## Intro Meeting Part 2 - 22 March 2024

Attending: Igor, Matt, Xinyue, Willow

### Action Items
1. **Downloading the data**: Igor ran into a few permissions snags with downloading the data from TNS. Matt and Xinyue suggested WISeREP, and Xinyue demonstrated being able to download the data from there. Xinyue and Matt after the meeting will figure out what query to use when gathering the data so we have that for future reference. Igor or Xinyue will download the data.
2. **Hosting the data**: We figured that the resulting size of the dataset should be no more than 1GB so we will host the data here on GitHub. Willow suggested using the parquet file format (with Python Pandas and PyArrow) which is a binary file format that will save space as well as read/writing time compared to csv files.
3. **Future meetings**: Xinyue and Willow will plan to meet weekly to discuss the coding aspects of the project. The entire group will meet once every two weeks in order to direct the flow of the project.
4. **Beginning coding**: Once any amount of data is uploaded to the GitHub repo, Xinyue and Willow can begin coding a preliminary autoencoder.

---

## Intro Meeting Part 1 - 21 March 2024

Attending: Federica, Igor, Matthew Graham, Xinyue, Willow

### Action Items
1. **Downloading the data**: Igor will take the lead on downloading data from TNS. Depending on the complexity of the task or download bandwidth limitations, we will help him.
2. **Hosting the data**: Depending on the size of the raw data, it may be hosted here on GitHub (if less then ~500MB), on Google Drive (if it can fit in one of our GDrives), or we can get an account on Google Gemini which givbes 2TB of storage.
3. **Future meetings**: We propose to meet every two weeks on Zoom.

### Raw Notes
* There is a historical aspect to this. Matthew says that Zwicky was the person who originally started creating new SN classes for basically everything he saw. Matthew says he remembers SNe types not just I and II but up to type 6 in Zwicky's work. He liked to put things in boxes if possible, and now we are stuck with that idea.
* Fed: The idea is for this project to be rather quick. The practical approach is clear: we build some sort of NN (prob an autoencoder) that produces some sort of latent space. Then we do an unsupervised exploration of the latent space to see how things group together. Then we compare this to current classification and then compare it to Zwicky's historical classifications. A potential addition to this work would be to add in more features to the dataset, such as host galaxy spectra or whatever. This could be a route for extending the project but not a priority.
* Igor: The first idea is to, for right now, just stick to one instrument rather than dealing with degrading and such. Will stick with SEDM.
* Fed: Should we also do this with a sample that is high resolution to see if we get the same clustering in latent space?
* Matthew: If we have spectra and lcvs. We could have high res and low res spectra. We could also degrade lcvs to be low cadence as well. Then: how does the classification work with high/low res both in classification and in spectra.
* Fed: Maybe we start with SEDM, then paper 2 once we are finished we could consider looking at high res spectra/lcv vs low res spectra/lcv.
* Igor: What I can do in the meantime: first step get all the public spectra on TNS (Transient Name Server) and get some statistics.
* **So Igor will do the data gathering but will just need the time for that.** This will take some time because getting the data is inherently time consuming and we can help Igor once he assess the difficulty of the task.
* Where will the data be? Igor likes Google Drive. 
* Igor: Talking about biases. We should consider making the analysis blind! We will download the official classification but make now plots or confusion matrices with the official true classification names until we are ready.
* Igor: There will also be a discrepancy between high and low SNR spectra. Sometimes it is hard to tell high and low SNR spectra apart.
* Matthew: NN should be able to deal with that.
* Fed: Autoencoder should save features and throw away the noise. 
* Matthew: If we run into this problem, we can look into a preprocessing step for this.
* Sherry: Maybe we can give different weights when we are training them.
* Fed: Let's do it iteratively. Let's first not be worried about SNR and then worry about it after we get started. 
* Igor: The SNR problem can be a huge bias so we should all do well to keep this in mind. The powerful part of this project is to be unbiased so we must be sure to be careful.
* Fed: The biases will be correlated with SNe type and SNe brightness/distance
* Sherry: Also the dataset will be extremely unbalanced by what we consider to be SN Ia
* Igor: What will the outliers be? For later on, this will be an interesting question :D
* Matthew: I think we decided not to use rest-frame for the spectra because there is the issue of not knowing the redshift.

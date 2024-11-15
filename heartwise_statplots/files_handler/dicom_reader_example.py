import numpy as np
import matplotlib.pyplot as plt
from heartwise_statplots.files_handler import DicomReader

# Define the DICOM file path
dicom_file_path = 'anonymous_ecg.dcm'

# Load the DICOM file
dicom = DicomReader.read_dicom_file(dicom_file_path)

# Extract diagnosis as a string
diagnosis = DicomReader.extract_diagnosis_from_dicom(dicom)
print(diagnosis)

# Extract ECG as a numpy array
leads_array = DicomReader.extract_ecg_from_dicom(dicom)

# Create a time axis in seconds
time_axis = np.arange(2500) / 250
leads_waveforms = leads_array[:, ::leads_array.shape[1] // 2500]
lead_i_waveform = leads_waveforms[0]

# Create a figure for Lead I
plt.figure(figsize=(15, 5))

# Plot Lead I
plt.plot(time_axis, lead_i_waveform)

# Add title and labels
plt.title('ECG Lead I (Einthoven) - Second Waveform', fontsize=16)
plt.xlabel('Time (s)', fontsize=14)
plt.ylabel('Amplitude (µV)', fontsize=14)

# Add grid for better readability
plt.grid(True)

# Adjust layout
plt.tight_layout()

# Save the plot as a PNG file without displaying it
output_filename = 'lead_i_ecg_second_waveform.png'
plt.savefig(output_filename, dpi=300)

# Close the plot to free memory
plt.close()

print(f"Lead I ECG plot from the second waveform saved as '{output_filename}'.")
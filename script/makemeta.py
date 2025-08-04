def Makemetadat(file_path):
  """ This function is make metadata from GEO matrix txt file """
  # Read lines
  with open(file_path, "r") as f:
    lines = f.readlines()

  # Keep only relevant metadata !Sample lines
  meta_lines = [line.strip() for line in lines if line.startswith("!Sample")]
  needed_lines = ["!Sample_geo_accession", "!Sample_title", '!Sample_characteristics_ch1']
  meta_lines = [line for line in meta_lines if any(need in line for need in needed_lines)]

  # Parse metadata into dictionary
  metadata_dict = {}
  characteristics_all = []

  for line in meta_lines:
    #Making each line into a list
      parts = line.split('\t')
    # Taking index 0 as key
      key = parts[0].replace("!Sample_", "")
    # Taking index 1:len(parts) as value
      values = parts[1:]

    # Check for multiple characteristics_ch1 lines
      if key.startswith("characteristics_ch1"):
        # Making a nested list from nested dictionary
          characteristics_all.append(values)
      else:
          metadata_dict[key] = values

  # Extract 'visit' and 'response' from characteristics
  # Assume characteristics_all[0] = visit, characteristics_all[1] = response
  visit_series = pd.Series(characteristics_all[0])
  response_series = pd.Series(characteristics_all[1])

  # Use regex to extract the values
  metadata_dict["visit"] = visit_series.str.extract(r'visit \(pre or on treatment\):\s*(\w+)', expand=False)
  metadata_dict["response"] = response_series.str.extract(r'response:\s*(\w+)', expand=False)

  # Convert to DataFrame
  metadata_df = pd.DataFrame(metadata_dict)

  # Clean up quotes from title and geo_accession
  metadata_df['title'] = metadata_df['title'].str.replace('"', '', regex=False)
  metadata_df['geo_accession'] = metadata_df['geo_accession'].str.replace('"', '', regex=False)

  return metadata_df


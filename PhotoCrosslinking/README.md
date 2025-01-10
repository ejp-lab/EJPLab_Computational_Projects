# Photo-Crosslinking Analysis Scripts for SILAC Pulldowns

*Authors*: Marshall Lougee

## kNN Imputation
**Description**: Imputes missing heavy/light values and removes proteins with quantification in less than 70% of replicates (line 70). Also computes P-Values for SILAC enrichment datasets using a one way t-test against a theoretical population mean of 1 (no enrichment). Note: check the distribution of data prior to conducting this analysis and make sure it is approximately normally distributed.

**Arguments**: Supply MaxQuant proteingroups.txt output file as the only argument

*example*: `python kNNimputation.py my_file.txt`

## Uniprot Annotation
**Description**: Annotates all proteins from minimalist filtering (i.e. kNN imputation) with UniProt domain and GO information

**Arguments**: Supply contaminant/imputed protein list (i.e. from kNN imputation) as only argument

*example*: `python UniProt_Annotation.py simplified_imputed.csv`

## Secondary Filtering GOclustering/DomainClustering
**Description**: Applies secondary filtering (cutoff) settings (i.e. fold-change/score/p-value) and conducts kmeans clustering based on GO terms or domains

**Arguments**: Minimal argument is the UniProt annotated hits file (i.e. from UniProt annotation). Alternative arguments can be viewed by running script with the argument -h

*example*: `python SecondaryFiltering_DomainClustering.py annotated_simplifed_imputed.csv -bh 0.01 -kmeans True -top_d 10`





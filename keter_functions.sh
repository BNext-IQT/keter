# Download and creates ChEMBL database using fancy piping technique where the dump file never touches
# the hard drive.
# chembl_to_postgres <version> <database_name>
# Example usage: chembl_to_postgres 27 chembl
function chembl_to_postgres {
    curl ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_${1}_postgresql.tar.gz | \
    tar xOz chembl_${1}/chembl_${1}_postgresql/chembl_${1}_postgresql.dmp | \
    pg_restore --no-owner -d ${2}
}
A blueprint for working with deep learning pipelines in science.

### Initial Setup

```
$ git clone https://jugit.fz-juelich.de/a.oberstrass/dl-project [NAME]
```
or 
```
$ git clone git@jugit.fz-juelich.de:a.oberstrass/dl-project.git [NAME]
```

Initialize the git submodules
```
$ cd [NAME]
$ git submodule update --init --recursive
```

Delete unnecessary files and folders and edit this README. To reset the commit history use
```
$ git remote remove origin
$ git remote add origin [NEW URL]
$ git reset $(git commit-tree HEAD^{tree} -m "Initial commit.")
$ git push -u origin --all
```

_Remove everything above and enter required information below._

# _Insert Project Name_

_Enter description here_


## Project structure


### Code 

`cli/`: All command line scripts belong here. All scripts should be written such that they can be executed from the root of the project, and are only using relative paths for portability.

`lab/`: All experimental code (notebooks etc.) belongs here. Once runnable it should be converted to cli programs in the folder cli/ for a more flexible use.

`code/`: Shared code between lab/ and cli/ but also between projects belongs here.

`env/`: This folder links to the [PLI JSC Environment](https://jugit.fz-juelich.de/aoberstrass/pli/pli-env) as submodule. The requirement files serve as a guideline for instantiating local Python environments. Adjustments can be made by either creating a new branch or detaching the submodule from the project.

### Data
`in/`: Contains links to datasets, raw data or data that were processed by other projects and are used as input for this project.

`data/`: Larger data that is being produced or stored along this project (e.g. trained models, processed data, ...) goes here. Files here are annexed and version controlled via DataLad.

`doc/`: All kinds of documentation such as charts, tables, graphics or conclusions that should persist belong here. Files stored here are stored in git itself and are not annexed.

`tmp/`: Project specific temporary files that do not need to be shared across siblings of this project go here. Rule of thumb: "Everything that can safely be deleted to save disk space at the end of the day belongs here". Contains standard folders for checkpoints (ckpt) and logs (log).

## Storage locations

|Name|Location|
|---|---|
|jsc| _/path/to/storage_ |


## Setup

This project uses [DataLad](https://www.datalad.org/) and [git annex](https://git-annex.branchable.com/) to manage stored files across various systems.

### Initialization
After cloning, initialize git annex with
```
$ git annex init DESC
```
where _DESC_ denotes a meaningful descriptor for the local repository. The data is typically stored on a clone of this repository with descriptor 'jsc' on the JSC filesystem.

Initialize the git submodules
```
$ git submodule update --init --recursive
```
You may then setup the virtual environment under `env/` following the README


### Adding remotes
You can add any remote clone of your new project and access the storage later with
```
$ git remote add REMOTE [path or ssh to remote]
```
Note that REMOTE has to fit the _DESC_ name for the remote repository (e.g. _jsc_ for the JSC filesystem) to be able to download files.


### Sync the repositories
To sync with the with all added remotes and the connected GitLab project use
```
$ git annex sync
```

### Get and receive files
Receive and push any file _FILE_ from and to any source _REMOTE_ with
```
$ git annex get FILE -f REMOTE
$ git annex copy FILE --to=REMOTE
```
List available remotes with
```
$ git annex whereis FILE
```

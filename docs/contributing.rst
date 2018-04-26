======================
Contributing to ndcube
======================

We are always enthusiastic to welcome new users and developers who
want to enhance the ndcube package.  You can contribute in
several ways, from providing feedback, reporting bugs,
contributing code, or reviewing pull requests.  There is a role for
almost any level of engagement.

Providing Feedback
------------------

We could always use more voices and opinions in the discussions about
ndcube and its development from both users and developers. There are a
number of ways to make your voice heard.  Whether it be constructive
criticism, inquiries about current or future capabilities, or
flattering praise, we would love to hear from you.  You can contact us
on the SunPy matrix channel or SunPy mailing lists.  See 
:ref:`getting_help`.

Reporting Bugs
--------------

If you run into unexpected behavior or a bug please report it. All
bugs are raised and stored on our `issue tracker`_.  If are not sure
if your problem is a bug, a deficiency in functionality, or something
else, send us a message on the SunPy chat room or an email to the
developer mailing list. Ideally, we would like a short code example so
we can run into the bug on our own machines. See :ref:`getting_help`.

.. _contributing_code:

Contributing Code
-----------------

If you would like to contribute code to ndcube, it is strongly
recommended that you first discuss your aims with the ndcube
community.  We aim to be an open and welcoming community for 
developers of all experience levels. Discussing your ideas
before you start can give you new insights that will make your
development easier, lead to a better end product, and reduce the
chances of your work being regetfully rejected because of an issue you
weren't aware of, e.g. the functionality already exists elsewhere.
See :ref:`getting_help` to contact the ndcube community.

Before you can contribute code to ndcube, you first need to install
the development version of ndcube.  To find out how, see
:ref:`dev_install`.  It is recommended that you perform the install
exactly as described as it will allow you to more easily follow the
typical workflow described below in :ref:`contributing_workflow`.

The ndcube code base is available and developed via the
`ndcube github repository`_.  If you are new to code development,
`github`_, or `git`_ (the version control software used by ndcube
developers) you can learn more from the following guides:
* `SunPy Newcomer's Guide`_
* `github guide`_;
* `git guide`_.
* `SunPy version control guide`_
The principles in the SunPy guides for contributing code and
utilizing GitHub and git are exactly the same for ndcube
except that we contribute to the ndcube repository rather than the
SunPy one.  If you need more help, ndcube and SunPy developers are
always happy to provide it.  See :ref:`getting_help`.  If you are a
more seasoned developer and would like to get more information, you
can check out the `SunPy Developer's Guide`_.

Proposed changes are submitted as a pull request (PR) via GitHub.
Once PRs are opened, it undergoes a process of code-review before it
can be approved and merged into the main repo.  Code review is a standard
industry practice which involves other members of the community reviewing 
your proposed changes and suggesting improvements.  It is a fantastic
way to improve your coding abilities as well as maintain common coding
standards throughout the package.  It is often also an effective
mechanism to get help with solving a problem.  A bugfix does not have
to be finished in order to open a PR.  In fact, most PRs are
incomplete when they are open.  This allows others to follow your
progress and contribute suggestions if you get stuck.  Anyone can review a
PR.  Experience is not a disqualifying factor.  But it is recommended
that at least one experienced developer reviews your code. Once you've
made the changes agreed upon by you and other developers, those
reviewing your code can approve the PR.  Approved PRs can then be
merged by those with write permissions to the repo.

.. _contributing_workflow:

Example Workflow for Contributing Code to ndcube
------------------------------------------------

In the rest of this section we will outline a simple workflow that
will allow you to get started with contributing to ndcube.  We will
assume you have installed the development version exactly as outlined
in :ref:`dev_install`.  Once that's done, open a new terminal.
Change into the directory containing the ndcube repository (repo) and
activate the conda environment in which the development version of
ndcube installed.  In Windows type:
.. code-block:: console

		$ activate ndcube-dev

In Linux or MacOS, type:
.. code-block:: console

		$ source activate ndcube-dev

First let's check what git branch we are on.  Git branches are a way
of having different versions of the same code within the repo
simultaneously. Assuming you have just installed the ndcube
development version, you will only have one branch, called master.  If
you have more, the * next to the branch name will indicate which
branch you are currently on. To check what branches you have and which
one you are on, type in the terminal:
.. code-block:: console

		$ git branch

If you are not on the master branch, let's change to it (referred to
as checking out) by typing:
.. code-block:: console

		$ git checkout master

Now, let's ensure we have the latest updates to the development
version.
.. code-block:: console

		$ git pull upstream master

This updates the local branch you are on (in this case, master) with
the version of the master branch stored in the "upstream" remote,
i.e. the original ndcube GitHub repo. 

At this point let's quickly talk about git remotes.  Remotes are
variables that point to URLs of GitHub repos.  In this example,
upstream is a remote pointing to the original ndcube GitHub
repo at https://github.com/sunpy/ndcube.  To see what remotes you have
attached to your local repo, and the URLs they point to, type:
.. code-block:: console

		$ git remote -v

If you have installed the ndcube development version as outlined in
:ref:`dev_install`, you will have two remotes:
* origin: https://github.com/my_github_handle/ndcube
* upstream: https://github.com/sunpy/ndcube
The :ref:`dev_install` instructions instruct you "fork" (copy between
GitHub accounts) the original ndcube repo to your own GitHub account.
You then "clone" it, i.e. copy the repo from GitHub to your local machine.
Therefore, the origin remote points where the local repo was "cloned"
from, i.e. the ndcube repo on your personal GitHub account.
The upstream remote, which the :ref:`dev_install` instructions
required you to add manually, points back to the main ndcube repo.
This enables you to get the latest updates as we did above.  The
remote names can be different depending on how you set them up so it's
a good idea to use the above command to confirm the names and URLs of
your remotes. Find out more about `git remotes`_ from the git online
guide.

Now, you are comfortable with git remotes, you are ready to start
coding!  Say you have found a bug in ndcube and would like to fix
it. As outlined above in :ref:`contributing_code`, it is strongly
recommended you talk to the ndcube community before you start coding
to get guidance on how and whether you should proceed.  Let's say
you've done that and have a clear plan on how to start.  The next task
is create a new git branch on which to make your changes.  This will allow
you to reserve your local master branch as a copy of the latest
upstream master branch. To create a new branch called my_fix, type:
.. code-block:: console

		$ git checkout -b my_fix

This will not only create the new branch but also check it out,
i.e. move you onto it. The new branch will now be an exact
copy of the branch from which you created it, in this case, the master
branch. But now you can edit files so that the new branch diverges
while keeping you master branch intact.

After a while, you've made some changes that partially or completely
fix the bug.  We now want to commit that change.  Committing is a bit
like saving except that it records the state of the entire code base.
You can then revert to this state at any time, even after new commits
have been made.  So if you mess up in the future you can always go
back to a version which worked.  This is why it is called version
controlling.  Before committing, we can see a list of files that we've
changed by typing:
.. code-block:: console

		$ git status

We can also get a summary of those changes, line by line:
.. code-block:: console

		$ git diff

Once we're happy with the changes, we must add the changed files to
the set of changed files to be included in the commit.  We do not have
to include all changed file.  We can add files one by one:
.. code-block:: console

		$ git add file1.py
		$ git add file2.py

or add all changed files at once:
.. code-block:: console

		$ git add --all

Be sure to check what files have changed before using this option to
make sure you know what you are committing.  Finally, to commit, type:
.. code-block:: console

		$ git commit

This will open a text editor, usually VI, and allow you to enter a
commit message to describe the changes you've made.  A commit message
is required before the commit can take place.  Once you've entered your
message, save it and exit your text editor.  Voila!  You've committed
your changes!!

To speed things up, the above process can be done in one command if
desired:
.. code-block:: console

		$ git commit -am 'My first commit.'

But CAUTION!  This adds and commits all changed files.  So make sure
you know what files have changed and how they've changed before doing
this.  Many a developer (inlcuding yours truly) have accidentally
committed extra files using this command and have had to spend wasted
time undoing their mistake.

Say it's the next day and you want to continue working on your bugfix.
Open a terminal, activate your ndcube dev conda environent, change
into the ndcube repo directory and make sure you are on the correct
branch.  Also make sure you pull any new updates on the upstream
master branch to your local bugfix branch:
.. code-block:: console

		$ source activate ndcube-dev (Just "activate ndcube-dev" in Windows)
		$ cd my_ndcube_repo
		$ git branch
		$ git checkout my_fix
		$ git pull upstream master

Assuming there are no updates that have caused conflicts with the
changes you made the other day, you're ready to continue working.

After more work and more commits, let's say you are ready to
issue a pull request (PR) to ndcube to get feedback on your work and
ultimately have it approved and merged! First you have to push your
changes to your GitHub account, using the origin remote:
.. code-block:: console

		$ git push origin my_fix

Now your changes are available on GitHub, follow these steps to open
a PR:
#. In a browser, go to your GitHub account and find your version of the git
   repo.  The URL should look like this:
   https://github.com/my_github_handle/ndcube/
#. There should be a green button on the right marked "Compare & pull
   request".  Click it.  If it is not there, click on the "Pull
   Requests" tab near the top of the page.  The URL should look like this:
   https://github.com/my_github_handle/ndcube/pulls
   Then click on the green "New Pull Request" button.  This will make
   a new page with four drop down menus appear near the top.
#. Set first drop down menu ("base fork") to "sunpy/ndcube" and the
   second one ("base") to "master".  This describes the repo and branch
   the changes are to be merged into.  Set the third drop down menu
   ("head fork") to "my_github_handle/ndcube" and the fourth
   ("compare") to "my_fix". This sets the repo and branch in which you
   have made the changes you want to merge.
#. Enter a title and a description of the PR in the appropriate
   boxes.  Try to be descriptive so other developers can understand
   the purpose of the PR.
#. Finally, click the green "Create Pull Request" button.  Well done!
   You've opened your first PR!

Now begins the process of code review, described above in
:ref:`contributing_code`.  You can make updates to your PR
based on suggestions from other members of the ndcube community by
editing your local my_fix branch, committing the new changes and
pushing them to you origin branch.  The PR will then be automatically
updated with the new commits.  Once the PR has been approved and
merged...Congratulations!  You have just contributed to ndcube!

Be sure to pull your the newly contributed changes to your local
master branch by doing:
.. code-block:: console

		$ git checkout master
		$ git pull upstream master

You are now ready to start using the newly improved development
version of ndcube, including your changes!

.. _issue tracker: https://github.com/sunpy/ndcube/issues
.. _ndcube github repository: https://github.com/sunpy/ndcube
.. _github: https://github.com/
.. _git: https://git-scm.com/
.. _SunPy Newcomer's Guide: http://docs.sunpy.org/en/stable/dev_guide/newcomers.html
.. _github guide: https://guides.github.com/
.. _git guide: https://git-scm.com/book/en/v2/Getting-Started-Git-Basics
.. _SunPy Developer's Guide: http://docs.sunpy.org/en/stable/dev_guide
.. _SunPy version control guide: http://docs.sunpy.org/en/stable/dev_guide/version_control.html
.. _git remotes: https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes

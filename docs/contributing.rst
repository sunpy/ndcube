======================
Contributing to ndcube
======================

We are always enthusiastic to welcome new users and developers who
want to enhance the ndcube package.  You can contribute in
several ways, from providing feedback, reporting bugs,
contributing code, and reviewing pull requests.  There is a role for
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
bugs are raised and stored on our `issue tracker`_.  If you are not
sure if your problem is a bug, a deficiency in functionality, or
something else, send us a message on the SunPy chat room or an email
to the developer mailing list. Ideally, we would like a short code
example so we can run into the bug on our own machines. See
:ref:`getting_help`.

.. _contributing_code:

Contributing Code
-----------------

If you would like to contribute code, it is strongly
recommended that you first discuss your aims with the ndcube
community.  We strive to be an open and welcoming community for 
developers of all experience levels. Discussing your ideas
before you start can give you new insights that will make your
development easier, lead to a better end product, and reduce the
chances of your work being regetfully rejected because of an issue you
weren't aware of, e.g. the functionality already exists elsewhere.
See :ref:`getting_help` to contact the ndcube community.

In the rest of this section we will go through the steps needed to set
up your system so you can contribute code to ndcube.  This is done
using `git`_ version control software and `GitHub`_,  a website that
allows you to upload, update, and share code repositories (repos).  If
you are new to code development or git and GitHub you can learn more
from the following guides:

* `SunPy Newcomer's Guide`_
* `github guide`_
* `git guide`_
* `SunPy version control guide`_

The principles in the SunPy guides for contributing code and
utilizing GitHub and git are exactly the same for ndcube
except that we contribute to the ndcube repository rather than the
SunPy one.  If you are a more seasoned developer and would
like to get further information, you can check out the `SunPy
Developer's Guide`_.

Before you can contribute code to ndcube, you first need to install
the development version of ndcube.  To find out how, see
:ref:`dev_install`.  In the rest of this section we will assume you
have performed the installation as described there.

Next, you will have to create a new online version of the ndcube
repo on your own GitHub account, a process known as "forking".  (If you
don't have a GitHub account, `sign up here`_.)  Sign into your GitHub
account and then go to the main `ndcube GitHub repository`_.  Click
the "Fork" button in the top right corner of the page.  A pop-up
window should appear asking to you to confirm which GitHub account you
wish to fork to.  Once you've done that, you should have a new
version of the ndcube repo on your own GitHub account.  It should
reside at a URL like https:/github.com/my_github_handle/ndcube.

Next, we need to link our newly forked online repo with the one we
created on our local machine as part of the installation.  To do
this, we will have to create a remote.  A `git remote`_ is an alias
pointing to the URL of a GitHub repo.  To see what remotes you have
and the URLs they point to, change into the local repo directory on
the command line and type:

.. code-block:: console

		$ git remote -v

If you have installed the ndcube development version as outlined in
:ref:`dev_install`, you will have one remote called ``origin`` pointing to
https://github.com/sunpy/ndcube.  Let's now add a remote to the repo
in your GitHub account called ``my_repo``.  In a terminal, from the local
repo directory, type:

.. code-block:: console

		$ git remote add my_repo https:/github.com/my_github_handle/ndcube

where you replace ``my_github_handle`` with your GitHub name.  Now you
can check that the remote has been added by again typing ``git remote -v``.

Now you're ready to get coding!  The following subsection will outline
an example workflow for contributing to ndcube.

.. _contributing_workflow:

Example Workflow for Contributing Code
--------------------------------------

To make changes to the development version of ndcube, we must first
activate the environment in which it is installed.  Recall during
installation, we named this environment ``ndcube-dev``.  From any
directory on the command line, Windows users should type:

.. code-block:: console

		> activate ndcube-dev

while Linux and MacOS users should type:

.. code-block:: console

		$ source activate ndcube-dev

Next, change into the local ndcube repo directory, ``ndcube-git``.
When you are making changes to ndcube, it is strongly recommended that
you use a different `git branch`_ for each set of related new features
and/or bug fixes. Git branches are a way of having different
versions of the same code within the repo simultaneously. Assuming you
have just installed the ndcube development version, you will only have
one branch, called ``master``.  It is recommended you do not do any
development on the ``master`` branch, but rather keep it as an clean copy
of the latest ``origin master`` branch.  If you have more than one
branch, the ``*`` next to the branch name will indicate which branch you
are currently on. To check what branches you have and which one you
are on, type in the terminal:

.. code-block:: console

		$ git branch

If you are not on the ``master`` branch, let's start by changing to it
(known as checking out the branch):

.. code-block:: console

		$ git checkout master

Now, let's ensure we have the latest updates to the development
version from the main repo.

.. code-block:: console

		$ git pull origin master

This updates the local branch you are on (in this case, ``master``) with
the version of the ``master`` branch stored in the ``origin`` remote,
i.e. the original ndcube GitHub repo.

Let's now create a new branch called ``my_fix`` on which to develop
our new feature of bugfix.  Type:

.. code-block:: console

		$ git checkout -b my_fix

This will not only create the new branch but also check it out. The
new branch will now be an exact copy of the branch from which you
created it, in this case, the ``master`` branch. But now you can edit
files so that the ``my_fix`` branch diverges while keeping your ``master``
branch intact.

After a while, you've made some changes that partially or completely
fix the bug.  We now want to commit that change.  Committing is a bit
like saving except that it records the state of the entire code base,
not just the file you've changed. You can then revert to this state at
any time, even after new commits have been made.  So if you mess up in
the future you can always go back to a version thats worked.  This is
why it's called version controlling.  Before committing, we can see a
list of files that we've changed by typing:

.. code-block:: console

		$ git status

We can also get a summary of those changes, line by line:

.. code-block:: console

		$ git diff

Once we're happy with the changes, we must add the changed files to
the set to be included in the commit.  We do not have
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

where ``'My first commit.'`` is the commit message.  But CAUTION!
This adds and commits all changed files.  So make sure you know what
files have changed and how they've changed before doing this.  Many a
developer has accidentally committed extra files using this command
and has wasted time undoing their mistake.

Say it's the next day and you want to continue working on your bugfix.
Open a terminal, activate your ``ndcube-dev`` conda environent, change
into the ``ndcube-git`` directory and make sure you are on the correct
branch.  Also make sure you pull any new updates from the ``origin``
``master`` branch to your local ``my_fix`` branch:

.. code-block:: console

		$ source activate ndcube-dev # For Windows users, type "activate ndcube-dev"
		$ cd ndcube-git
		$ git branch
		$ git checkout my_fix
		$ git pull origin master

Assuming there are no updates that conflict with the changes you made
the other day, you're ready to continue working.  If there are
conflicts, open the affected files and resolve them.

After more work and more commits, let's say you are ready to
issue a pull request (PR) to get feedback on your work and
ultimately have it approved and merged into the main repo! First you
have to push your changes to your GitHub account, using the ``my_repo``
remote:

.. code-block:: console

		$ git push my_repo my_fix

Now your changes are available on GitHub.  Follow the steps below to open
a PR:

#. In a browser, go to your GitHub account and find your version of the git
   repo.  The URL should look like this:
   https://github.com/my_github_handle/ndcube/
#. There should be a green button on the right marked "Compare & pull
   request".  Click it.  If it is not there, click on the "Pull
   Requests" tab near the top of the page.  The URL should look like this:
   https://github.com/my_github_handle/ndcube/pulls.
   Then click on the green "New Pull Request" button.  This will open
   a new page with four drop-down menus near the top.
#. Set the "base fork" drop-down menu to "sunpy/ndcube" and the
   "base" drop-down to "master".  This describes the repo and branch
   the changes are to be merged into.  Set the "head fork" drop-down
   menu to "my_github_handle/ndcube" and the "compare" drop-down to
   "my_fix". This sets the repo and branch in which you have made the
   changes you want to merge.
#. Enter a title and a description of the PR in the appropriate
   boxes.  Try to be descriptive so other developers can understand
   the purpose of the PR.
#. Finally, click the green "Create Pull Request" button.  Well done!
   You've opened your first PR!

Now begins the process of code review.  Code review is a standard
industry practice which involves other members of the community
reviewing your proposed changes and suggesting improvements.  It is a
fantastic way of improving your coding abilities as well as preserving
the integrity of the overall package.  A bugfix does not have
to be finished in order to open a PR. In fact, most PRs are incomplete
when they are first opened. This allows others to follow your progress
and contribute suggestions if you get stuck.  Anyone can review a  PR.
Experience is not a disqualifying factor.  But it is recommended that
at least one experienced developer reviews your code. You can make
updates to your PR by editing your local ``my_fix`` branch, committing
the new changes and pushing them to the ``my_repo`` remote.  The PR
will then be automatically updated with the new commits.  Once you've
made all changes and the online tests have passed, those reviewing
your code can approve the PR.  Approved PRs can then be merged by
those with write permissions to the repo.  Congratulations!  You have
just contributed to ndcube!

Be sure to pull your the newly contributed changes to your local
master branch by doing:

.. code-block:: console

		$ git checkout master
		$ git pull origin master

You are now ready to start using the newly improved development
version of ndcube, including your changes!

If you have questions about this guide or while making contributions,
ndcube and SunPy developers are always happy to help.  See
:ref:`getting_help`.  Happy coding and talk to you soon!

.. _issue tracker: https://github.com/sunpy/ndcube/issues
.. _sign up here: https://github.com/join?source=header-home
.. _ndcube GitHub repository: https://github.com/sunpy/ndcube
.. _GitHub: https://github.com/
.. _git: https://git-scm.com/
.. _SunPy Newcomer's Guide: http://docs.sunpy.org/en/stable/dev_guide/newcomers.html
.. _github guide: https://guides.github.com/
.. _git guide: https://git-scm.com/book/en/v2/Getting-Started-Git-Basics
.. _SunPy version control guide: http://docs.sunpy.org/en/stable/dev_guide/version_control.html
.. _SunPy Developer's Guide: http://docs.sunpy.org/en/stable/dev_guide
.. _pull requests: https://help.github.com/articles/about-pull-requests/
.. _git branch: https://git-scm.com/book/en/v2/Git-Branching-Branches-in-a-Nutshell
.. _git remote: https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes

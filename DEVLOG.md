simple-cuda devlog
================== 





7-2-2014: Removing Compile.hs from list of modules. This module takes
	  implements the compilation of an ACC AST. 
	  Need to replace this with a SimpleACC compiler.
 
	  Removind CUDA.hs from list of modules. This module will 
	  be replaced by a SimpleCUDA module that implements a SimpleBackend
	  instance for CUDA. 

	  Removing the Execure modules from list. These needs to 
	  be replaced with similar modules but tweaked for BackendClass 
	  usage. 

	  Started a SimpleCUDA module that houses a SimpleBackend instance.


7-1-2014: redoing most changes from the bkit branch into a bkit0-15
	  branch (that should be up to date with accelerate-cuda 0.15). 
	  


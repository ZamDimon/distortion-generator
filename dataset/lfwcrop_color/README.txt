LFWcrop is a cropped version of the Labeled Faces in the Wild (LFW)
dataset, keeping only the center portion of each image (i.e. the face).
In the vast majority of images almost all of the background is omitted.

LFWcrop was created due to concern about the misuse of the original
LFW dataset, where face matching accuracy can be unrealistically
boosted through the use of background parts of images (i.e.
exploitation of possible correlations between faces and backgrounds).

For each LFW image, the area inside a fixed bounding box was extracted.
The bounding box was at the same location for all images, with the
upper-left and lower-right corners being (83,92) and (166,175),
respectively. The extracted area was then scaled to a size of 64x64
pixels. The selection of the bounding box location was based on the
positions of 40 randomly selected LFW faces [1].

As the location and size of faces in LFW was determined through
the use of an automatic face locator (detector), the cropped faces
in LFWcrop exhibit real-life conditions, including mis-alignment,
scale variations, in-plane as well as out-of-plane rotations.

The "faces" directory contains all of the cropped faces,
i.e. there are no sub-directories for each person.

The "lists" directory contains text files which describe the assignment
of faces into sets, including the split into training and testing
subsets within each set. The text files follow the LFW protocol [2].
The "diff" suffix indicates mismatched pairs, where each line contains
the names of images of two different people. The "same" suffix indicates
matched pairs, where each line contains the names of images of the same
person.


References

[1] C. Sanderson, B.C. Lovell.
    Multi-Region Probabilistic Histograms for Robust
    and Scalable Identity Inference.
    ICB 2009, LNCS 5558, pp. 199-208, 2009.

[2] G.B. Huang, M. Ramesh, T. Berg, E. Learned-Miller.
    Labeled Faces in the Wild: A Database for Studying
    Face Recognition in Unconstrained Environments.
    University of Massachusetts, Amherst, 
    Technical Report 07-49, 2007.

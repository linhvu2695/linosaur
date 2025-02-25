---
date: 2020-05-20 10:56:15
layout: post
title: Microchip - putting Paris in a bottle
subtitle: A whole city architectured to fit inside a piece of glass with the size of less than a fingernail
description: A chip is dead. Who killed the chip? Travel with the forensic team and have a look inside the organs of a Microchip, to see the anatomy of the greatest invention of humankind in the 20th century. A brain that is made of silicon, size of a fingernail and strength of a billion transistors.
image: https://www.student-circuit.com/wp-content/uploads/sites/54/2023/11/AdobeStock_592529535.jpeg
optimized_image: https://www.student-circuit.com/wp-content/uploads/sites/54/2023/11/AdobeStock_592529535.jpeg
category: semiconductor
tags:
  - semiconductor
author: linhvu2695
paginate: true
---
Microchip is undeniably the greatest invention of humankind since the end of the Atomic Era. It has been the fuel for the current Information Era, the backbone of modern technology, and is now everywhere in the daily life of people. Most of us have seen the image of a microchip: a little black square with a size slightly larger than a coin that sits silently in the corner of our laptops, PCs, or smartphones. But few have ever seen the very inside of these black boxes, where a whole city of glass and metal was built, a miraculous feat of engineering that realizes the idea of “given enough Ifs, I can put the whole Paris in a bottle”.
<img src="https://images.idgesg.net/images/article/2018/12/snapdragon-855-mobile-platform-chip-compaison-us-coin-2-100782075-orig.jpg">
*Snapdragon 855 – the CPU that runs Samsung Galaxy S10 in comparison with a one-cent coin.*
I am an engineer at Qualcomm – the company that made that little Snapdragon 855 you see in the above picture (Fig.1). My daily work is to perform surgery on the dead bodies of unfortunate microchips, those who did not survive the production process, or were killed when being used by “cruel” users. And just like in CSI: Miami, I’m a member of the Forensic team who would open up these corpses, look into their organs & skins & bones, and try to find out the reason of death. Join me in one of these surgery sessions, where you get to see a microchip in the way you’ve never seen before, demystifying all questions you used to have on those mysterious black boxes.

# Decapsulation
Microchips came in different sizes, shapes, and packages. However, their structure is basically the same. In the below picture, you can have a brief idea of what is inside the chip as the surgeon started to open up its body and reveal its organs – a process we called decapsulation. Inside that black box, positioned in the center of everything, is the **silicon die**, the most sacred, important part of the whole chip. If compare the microchip to a human body, then the silicon die is definitely the brain, a super big brain that takes up almost all the space, which makes sense since all the chip ever does is computing.
<img src="https://res.cloudinary.com/dptj6j9y9/image/upload/v1740499371/decap-2_z54dbf.png">
*Decapsulation of a QFP (Quad-Flat-Package) ATmega32U4 with 44 pins (11 pins on each side).*

Besides being very important, the die is also extremely fragile (silicon is actually glass), and therefore must be protected by wrapping around it a plastic resin – called **mold compound**. This compound is the black material that you see embracing the die and wires, keeping them safe inside an enclosed capsule (now you see why opening it up is called de-capsulation). Of course, it cannot fully enshroud the die – there must be a way for the die to make contact with the outside world. And that connection is made using **bonding wires** – each one connects from the die to a **pin** outside. These bonding wires are previously made of actual gold – some Youtubers really go collect these thrown away devices and extract the gold from them, though it requires professional chemistry level. The outside pins will later be soldered into the motherboard – spreading electrical signals from the die to the other devices.

# Silicon Die
Now comes the exciting part – to investigate the silicon die. After carefully removing the die from its protective mold compound, we placed it under the microscope and take a look – as in below image is showing you (you can still see the aforementioned bonding wires attached to several pads along the borders).
<img src="https://res.cloudinary.com/dptj6j9y9/image/upload/v1740499508/Capture_i4293l.png">
*Overview of a silicon die (left) & High magnification of the integrated circuits (right)*

If the black microchip is already very small, this die is even smaller, and thinner.  There are reasons as to why the silicon die becomes the most valuable component in the chip, the most simple and straightforward to be – it is incredibly complex. Despite its tiny size, the die is literally a city with houses & buildings made of transistors, connecting together by streets of copper and highways of aluminum creating multiple layers of metal. The latest microprocessors often consist of more than 10 **metal layers**, each one stacking on top of each other and are connected together using **metal vias**. This system of metal lines and cylindrical contacts is called layers of **interconnects**.

A bird-eye view is presented to you in above image. You can see different levels of metal traces crossing each other with more blurry layers on top. The 3D images below describe how these metal layers are stacked up and numbered accordingly.
<img src="https://res.cloudinary.com/dptj6j9y9/image/upload/v1740499508/Capture1-1_bkxgya.png">
*Cross-section of the multiple metal layers in a silicon die*

Below all these metal traces and vias are the layer of the famous **transistors**. While the metal traces only help in connection, transistors are the main components that actually compute. These transistors are similar to neurons in human brains. Even though each one of them basically just controls a single 0 or 1 value (called boolean value), its strength lies in the massive numbers. The latest Apple A13 Bionic consists of around 8.5 billion transistors, more than the population of current Earth. And to be able to pack billions of transistors on an area slightly larger than a fingernail, each one of them has to be shrunk to the size of a 10 – 20 nanometer wide. How small is that? Small enough to pack more than one-thousand transistors across the diameter-of-your-hair. This struggle to shrink transistor size has been going on for decades, with every 2 years the chipmakers have been able to make them two times smaller than before, a journey that is often known to us as Moore’s law.
<img src="https://res.cloudinary.com/dptj6j9y9/image/upload/v1740499621/Picture1-1_l7ovuc.png">
*Cross-section of IC die showing metal layers (back-end) & transistors (front-end)*

Finally, below the transistor level is a thick layer of silicon, which is called **substrate**. This layer contains all the doping necessary to make the transistors active. The material for substrate must always be semiconductor – able to switch between conducting and non-conducting states – enabling the transistor to represent a binary value of either 0 or 1. Almost every current microchip is using silicon (Si) as its substrate element, due to its efficiency and remarkably low-cost. The silicon was manufactured from sand, not the one that can be found from beaches, but silica sand which people usually get from quarrying. This high-quality sand will later be refined into pure Silicon with an exceptional low level of foreign particles, and then be used in foundries to create the substrate for microchips.

And that is brief anatomy of what is inside a microchip. To finalize this article below is some fancy video footage from Intel on how microchips are made. The technology in the clip is 22nm, which is a few years back, but its visual illustration is still very interesting and useful to grab an idea of how this Paris in a bottle was created.
<iframe width="744" height="419" src="https://www.youtube.com/embed/d9SWNLZvA8g" title="Intel: The Making of a Chip with 22nm/3D Transistors | Intel" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
*Intel: The Making of a Chip with 22nm/3D Transistors*
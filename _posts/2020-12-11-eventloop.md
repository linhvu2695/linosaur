---
date: 2020-12-11 00:26:35
layout: post
title: Event Loop - one thing at a time
subtitle: Here's a function. Call me maybe?
description: Here's a function. Call me maybe?
image: https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fz9mcu9ytmluzti4khdv2.png
optimized_image: https://media2.dev.to/dynamic/image/width=800%2Cheight=%2Cfit=scale-down%2Cgravity=auto%2Cformat=auto/https%3A%2F%2Fdev-to-uploads.s3.amazonaws.com%2Fuploads%2Farticles%2Fz9mcu9ytmluzti4khdv2.png
category: software
tags:
  - software
  - javascript
author: linhvu2695
paginate: true
---
So let's talk about Event Loop, the thing they always ask you in interviews. You can either read this article, or a better choice is to watch this video with more than 3.5M views (for a programming YouTube clip!). This talk is the perfect example of "If you can't explain it simply, you don't understand it well enough". Well done Philip.
<iframe width="1212" height="354" src="https://www.youtube.com/embed/8aGhZQkoFbQ" title="What the heck is the event loop anyway? | Philip Roberts | JSConf EU" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

# Why asynchronous non-blocking I/O?
Details in <a href="http://www.kegel.com/c10k.html">C10K problem</a>. Shortly speaking, OS thread is costly (memory, CPU time, context switching). Models with 1 thread / 1 request is no longer effecient. 

NodeJS is a single threaded application to help execute JavaScript code on server side. All the code are executed on just one single thread. Then how can it serves thousands of requests at a time? The answer is **non-blocking asynchronous I/O**. Instead of running a heavy function and wait for it to finish, NodeJS will simply dispatch that task to OS kernel or other workers and continue processing the main thread, staying responsive and free from freezing.

# How does Event Loop work?
Exactly like its name, Event Loop is an infinite loop that runs forever in Javascript Runtime (V8 in Google Chrome) to listen to events.

<img src="https://media.licdn.com/dms/image/v2/D5612AQE4eKWf_kLA5g/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1676921600016?e=2147483647&v=beta&t=G5JqfRxFXShpYagJ4_bgZmOX-hdokwgkC-4qpgH4nfY">

Event Loop job is very simple: it will read the **Stack** & **Event Queue**. If the Stack is empty, it will pick the first event in the queue, along with the attached **Hanlder** (callback or listener) and pushed into the stack. When you click a button, an Event is created and pushed into the Event Queue.

```javascript
console.log('Start');

setTimeout(() => {
  console.log('Async operation');
}, 2000);

console.log('End');
```
In the above example, "Start" & "End" will be logged first, while "Async operation" will only appear after 2 seconds.

## WebAPIs
In JavaScript and web browsers, Web APIs refer to sets of functionalities provided by the browser environment to interact with web-related features and resources. These APIs include the Document Object Model (DOM) API for manipulating HTML and CSS, the XMLHttpRequest (XHR) or Fetch API for making HTTP requests, the Geolocation API for accessing user location information, the localStorage and sessionStorage APIs for storing data locally in the browser, and many others.

## Useful patterns
There are some popular patterns that are used commonly in JS to utilize the asynchronous capability

1. Execute a function in async
    ```javascript
    importantFunction()

    setTimeout(lessImportantFunction() , 0);

    moreImportantFunction()
    ```
2. Chunking: process a large array of data by chunks, each one after another, on a separate thread. Of course the below code needs some extra effort to handle the case where we reach the end of the array, but you get the idea.
    ```javascript
    function processArray(bigArr, start) {
        for (let i = start; i < BATCH_SIZE; i++) {
            heavyProcess(bigArr[i]);
        }
        setTimeout(() => processArray(bigArr, start + BATCH_SIZE), 0);
    }
    ```

## Event Queues
<img src="https://media.licdn.com/dms/image/v2/D5612AQHIuZDc3cqPtg/article-cover_image-shrink_600_2000/article-cover_image-shrink_600_2000/0/1721189705579?e=2147483647&v=beta&t=7z1ivEBMlIOpeq4P2UUbbrj1T64ysIpkPv27efVvq60">
Microtasks are usually promises and mutation observers. When a promise resolves, its `.then()` callback goes to the microtask queue. Microtasks are the top priority. After finishing a task in the call stack, the event loop first handles all tasks in the microtask queue before moving to the macrotask queue, ensuring crucial operations are done quickly.

Macrotasks include timers (setTimeout, setInterval), I/O operations, and other events. These tasks go into the macrotask queue. The event loop processes them one by one in a first-in, first-out (FIFO) order, but only after the call stack and microtask queue are empty.

Understanding these core concepts is crucial for building efficient and responsive applications. Keep exploring and happy coding! 